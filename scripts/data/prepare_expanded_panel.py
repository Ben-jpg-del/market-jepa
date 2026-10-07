"""Build the authorized historical IWB proxy panel with explicit coverage."""
from pathlib import Path
import concurrent.futures, getpass, hashlib, json, sys, threading, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data import FMPClient,DataAccessError,fetch_price_history,load_panel,save_panel,build_panel,monthly_universe,DateBatchSampler
from src.expanded_data import IWBHistoryClient,fetch_monthly_history,save_roster_bundle,load_roster_bundle,membership_from_snapshots
from src.cloud_data import _source_fingerprint
from src.features import REFERENCE_ETFS,rolling

OUTPUT=ROOT/'work/fjepa-expanded-data'
CACHE=ROOT/'work/fmp-live-cache'
START='2003-01-01';END='2021-12-31'
SYMBOL_MAP={'BRKB':'BRK-B','BF/B':'BF-B','BF.B':'BF-B'}
PARSER_SHA256=hashlib.sha256((ROOT/'src/expanded_data.py').read_bytes()).hexdigest()
ROSTER_POLICY_VERSION=2

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def cached_request(endpoint,params):
    identity=json.dumps({'endpoint':endpoint,'params':params},sort_keys=True)
    return (CACHE/(hashlib.sha256(identity.encode()).hexdigest()+'.json')).exists()

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    original=load_panel(ROOT/'work/fjepa-cloud-data/panel_raw.npz')
    dates=original.dates[original.dates<=np.datetime64(END)]
    roster=OUTPUT/'iwb_rosters.json'
    snapshots,roster_report=load_roster_bundle(roster) if roster.exists() else (None,{})
    if (roster_report.get('validated_format_aliases')!=SYMBOL_MAP or
        roster_report.get('parser_sha256')!=PARSER_SHA256 or roster_report.get('roster_policy_version')!=ROSTER_POLICY_VERSION):
        print('Downloading and validating archived monthly IWB holdings.',flush=True)
        client=IWBHistoryClient(ROOT/'work/iwb-history-cache',symbol_map=SYMBOL_MAP)
        try:
            snapshots,roster_report=fetch_monthly_history(client,dates,allow_internal_gaps=True)
        except Exception as exc:
            if hasattr(exc,'coverage_report'):write_json(OUTPUT/'roster_failure.json',exc.coverage_report)
            raise
        roster_report['validated_format_aliases']=SYMBOL_MAP
        roster_report['parser_sha256']=PARSER_SHA256
        roster_report['roster_policy_version']=ROSTER_POLICY_VERSION
        excluded=[]
        for snapshot in snapshots:
            for holding in snapshot.holdings:
                if holding['name'] in {'LIBERTY VENTURES RTS','LIBERTY MEDIA RIGHTS EXP JUNE COR','T MOBILE US SUBSCRIPTION RIGHTS IN'}:
                    excluded.append({'date':snapshot.as_of_date,'symbol':holding['symbol'],'name':holding['name']})
                    holding['symbol']=None
        roster_report['excluded_identified_rights']=excluded
        save_roster_bundle(snapshots,roster_report,roster)
    symbols,membership,timing=membership_from_snapshots(dates,snapshots)
    report={'status':'downloading','universe':roster_report,'timing':timing,'start':START,'end':END,
            'point_in_time_verified':False,'max_stocks':1000,'candidate_count':len(symbols),
            'limitations':roster_report['limitations']+['Terminal delisting returns unavailable; no investable alpha claim.']}
    write_json(OUTPUT/'coverage.json',report)
    print(json.dumps({'candidate_symbols':len(symbols),'first_snapshot':roster_report['first_snapshot'],'valid_months':roster_report['valid_months']}),flush=True)
    key=getpass.getpass('FMP API key (hidden): ')
    gate=threading.Lock();next_request=[0.0];local=threading.local()
    class LimitedClient(FMPClient):
        def get(self,endpoint,params=None,refresh=False):
            if refresh or not cached_request(endpoint,dict(params or {})):
                with gate:
                    wait=next_request[0]-time.monotonic()
                    if wait>0:time.sleep(wait)
                    next_request[0]=time.monotonic()+.25
            return super().get(endpoint,params,refresh)
    def fetch(symbol):
        if not hasattr(local,'client'):local.client=LimitedClient(CACHE,api_key=key)
        frames=[];errors={}
        for adjusted in (True,False):
            endpoint='historical-price-eod/'+('dividend-adjusted' if adjusted else 'non-split-adjusted')
            oldparams={'symbol':symbol,'from':START,'to':'2025-12-31'}
            end='2025-12-31' if cached_request(endpoint,oldparams) else END
            try:frames.append(fetch_price_history(local.client,symbol,START,end,adjusted).loc[START:END])
            except DataAccessError as exc:
                errors['adjusted' if adjusted else 'raw']=exc.as_dict()
                if adjusted:break
        if not frames:return symbol,None,errors
        return symbol,frames[0].join(frames[1],how='left') if len(frames)==2 else frames[0],errors
    names=list(dict.fromkeys([*symbols,'SPY',*REFERENCE_ETFS]));prices={};failures={};done=0;last_progress=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for symbol,frame,errors in pool.map(fetch,names):
            done+=1
            if frame is not None:prices[symbol]=frame
            if errors:failures[symbol]=errors
            if time.monotonic()-last_progress>=15 or done==len(names):
                last_progress=time.monotonic()
                report['downloads']={'completed':done,'requested':len(names),'adjusted_histories':len(prices),'failures':failures}
                write_json(OUTPUT/'coverage.json',report)
                print(json.dumps({'downloaded':done,'total':len(names),'usable_histories':len(prices),'symbols_with_failures':len(failures)}),flush=True)
    if any(s not in prices for s in ['SPY',*REFERENCE_ETFS]):raise RuntimeError('Missing required reference history')
    if not np.array_equal(prices['SPY'].index.to_numpy(dtype='datetime64[D]'),dates):raise RuntimeError('Benchmark calendar mismatch')
    empty=pd.DataFrame(columns=['Open','High','Low','Close','Volume','RawOpen','RawClose','RawVolume'],index=pd.DatetimeIndex([],name='date'),dtype=float)
    for s in symbols:
        if s not in prices:prices[s]=empty.copy()
    # Archived holdings can contain retrospectively renamed/reused symbols.
    # Require a contemporaneous source-price anchor, never a current profile.
    # Five percent is predeclared before any model/development scores are seen.
    lookup={s:i for i,s in enumerate(symbols)}
    chosen=np.asarray(timing['snapshot_date_by_origin'])
    anchor_records=[];anchor_counts={'matched':0,'mismatch':0,'missing_raw_close':0,'missing_source_price_or_non_usd':0}
    for snapshot in snapshots:
        origin_rows=chosen==snapshot.as_of_date
        day=pd.Timestamp(snapshot.as_of_date)
        for holding in snapshot.holdings:
            symbol=holding['symbol']
            if symbol is None:continue
            price=holding.get('source_price');currency=holding.get('market_currency')
            if currency in ('-',None,'') and holding.get('currency')=='USD' and holding.get('source_fx_rate')=='1.00':currency='USD'
            raw=prices[symbol].get('RawClose',pd.Series(dtype=float)).get(day,np.nan)
            if price is None or not np.isfinite(price) or price<=0 or currency!='USD':reason='missing_source_price_or_non_usd'
            elif not np.isfinite(raw):reason='missing_raw_close'
            elif abs(raw/price-1)>.05:reason='mismatch'
            else:reason='matched'
            anchor_counts[reason]+=1
            if reason!='matched':
                membership[origin_rows,lookup[symbol]]=False
                anchor_records.append({'as_of_date':snapshot.as_of_date,'symbol':symbol,'source_name':holding['name'],
                    'source_price':price,'fmp_raw_close':float(raw) if np.isfinite(raw) else None,'reason':reason})
    report['price_anchor_validation']={'relative_tolerance':.05,'counts':anchor_counts,
        'note':'Membership requires contemporaneous raw close within5% of archived USD holdings price. This screens identity errors; it does not certify identifiers, corporate-action continuity or publication timing.',
        'exceptions_file':str(OUTPUT/'price_anchor_exclusions.json')}
    write_json(OUTPUT/'price_anchor_exclusions.json',anchor_records)
    print(json.dumps({'price_anchor_validation':report['price_anchor_validation']}),flush=True)
    calendar=pd.DatetimeIndex(dates)
    def matrix(column):return np.column_stack([prices[s].reindex(calendar)[column].to_numpy(float) if column in prices[s] else np.full(len(dates),np.nan) for s in symbols])
    raw_close,raw_volume=matrix('RawClose'),matrix('RawVolume')
    selected=monthly_universe(dates,membership,raw_close,raw_volume,max_stocks=1000)
    count=selected.sum(axis=1);train=(dates>=np.datetime64(timing['first_covered_date']))&(dates<=np.datetime64('2018-12-31'))
    report['selection']={'train_dates':int(train.sum()),'minimum_train_count':int(count[train].min()),'median_train_count':float(np.median(count[train])),
        'maximum_count':int(count.max()),'train_dates_under_128':int((count[train]<128).sum()),'train_dates_under_32':int((count[train]<32).sum())}
    if np.median(count[train])<256:raise RuntimeError('Expanded data coverage too small to support intended comparison')
    keep=selected.any(axis=0);kept=[s for s,k in zip(symbols,keep) if k]
    report['retained_symbols']=kept;report['retained_symbol_count']=len(kept)
    report['never_selected_symbols']=[s for s,k in zip(symbols,keep) if not k]
    report['stock_coverage']={s:{'adjusted_rows':int(prices[s].get('Close',pd.Series(dtype=float)).notna().sum()),
        'raw_rows':int(prices[s].get('RawClose',pd.Series(dtype=float)).notna().sum()),'member_dates':int(membership[:,i].sum()),'selected_dates':int(selected[:,i].sum())}
        for i,s in enumerate(symbols)}
    report['status']='computing_features';write_json(OUTPUT/'coverage.json',report)
    print(json.dumps({'computing_features_for':len(kept),'selection':report['selection']}),flush=True)
    source=_source_fingerprint(prices,symbols,membership)
    retained={s:prices[s] for s in [*kept,'SPY',*REFERENCE_ETFS]}
    panel=build_panel(retained,kept,membership=membership[:,keep],universe_mode='historical_candidate',max_stocks=1000)
    if not np.array_equal(panel.universe,selected[:,keep]):raise RuntimeError('Selection changed after pruning')
    roster_hash=hashlib.sha256(roster.read_bytes()).hexdigest()
    parent=panel.metadata['dataset_fingerprint']
    panel.metadata.update(universe_label='Historical IWB monthly holdings proxy',exact_russell_index_membership=False,
        source_fingerprint=source,roster_sha256=roster_hash,first_membership_date=timing['first_covered_date'],
        dataset_fingerprint=hashlib.sha256((parent+source+roster_hash).encode()).hexdigest(),point_in_time_verified=False)
    panel.metadata['limitations']+=report['limitations']
    panelpath=OUTPUT/'panel_raw.npz';save_panel(panel,panelpath)
    report.update(status='ready_research_only',dataset_fingerprint=panel.metadata['dataset_fingerprint'],source_fingerprint=source,
        roster_sha256=roster_hash,stock_shape=list(panel.stock.shape),market_shape=list(panel.market.shape),panel_path=str(panelpath),
        panel_sha256=hashlib.file_digest(panelpath.open('rb'),'sha256').hexdigest())
    report['samplers']={}
    complete=rolling(panel.stock_obs[:,:,:4].all(axis=-1).astype(float),60,'sum')==60
    eligibility=panel.universe & complete
    times=np.arange(len(panel.dates));cutoff=np.searchsorted(panel.dates,np.datetime64('2018-12-31'),side='right')-1
    for horizon in (1,5):
        valid=(times>=59)&(times+horizon<=cutoff)&(panel.dates>=np.datetime64('2006-01-01'))
        valid&=(eligibility.sum(axis=1)>=2)&panel.market_obs[:,0]
        report['samplers'][str(horizon)]={'origins':int(valid.sum()),'eligible_contexts':int(eligibility[valid].sum())}
    write_json(OUTPUT/'coverage.json',report)
    print(json.dumps({'status':report['status'],'panel_bytes':panelpath.stat().st_size,'selection':report['selection'],'samplers':report['samplers']}),flush=True)

if __name__=='__main__':main()
