"""Causality, accounting, credential safety and resumability tests for pilot data."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from src.data import (FMPClient,DataAccessError,DateBatchSampler,synthetic_panel,
    fit_normalizer,normalize_panel,save_panel,load_panel,monthly_universe,reconstruct_membership,
    build_panel,_prices_frame,fetch_price_history)
from src.features import compute_features,janus_score,reactivity,REFERENCE_ETFS


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(n_dates=850,n_stocks=6,start="2016-01-01")


def normalized(panel):
    return normalize_panel(panel,fit_normalizer(panel))


def test_feature_prefix_invariance():
    rng=np.random.default_rng(11);n=700;k=4
    close=100*np.exp(np.cumsum(rng.normal(0,.01,(n,k)),axis=0))
    open_=close*np.exp(rng.normal(0,.002,(n,k)))
    high=np.maximum(open_,close)*1.01;low=np.minimum(open_,close)*.99
    volume=rng.lognormal(14,.3,(n,k))
    spy=100*np.exp(np.cumsum(rng.normal(0,.008,n)))
    reference=100*np.exp(np.cumsum(rng.normal(0,.01,(n,9)),axis=0))
    eligible=np.ones((n,k),bool)
    inputs=(open_,high,low,close,volume,spy,reference,eligible)
    full=compute_features(*inputs)
    prefix=compute_features(*(x[:650] for x in inputs))
    for a,b in zip(full,prefix):
        np.testing.assert_allclose(a[:650],b,equal_nan=True,rtol=1e-9,atol=1e-9)


def test_fit_normalizer_ignores_future(panel):
    before=fit_normalizer(panel)
    altered=copy.deepcopy(panel)
    altered.stock[altered.dates>np.datetime64("2018-12-31")]=1e9
    altered.market[altered.dates>np.datetime64("2018-12-31")]=-1e9
    after=fit_normalizer(altered)
    assert before==after


def test_missing_future_does_not_change_origin_eligibility(panel):
    p=normalized(panel)
    original=DateBatchSampler(p,batch_dates=1,stocks_per_date=6)
    t=int(original.origins[-10]);altered=copy.deepcopy(p)
    altered.stock_obs[t+1,:,0]=False
    other=DateBatchSampler(altered,batch_dates=1,stocks_per_date=6)
    np.testing.assert_array_equal(original.eligibility[t],other.eligibility[t])
    original.origins=np.array([t]);other.origins=np.array([t])
    before=original();after=other()
    np.testing.assert_array_equal(before["symbol_indices"],after["symbol_indices"])
    np.testing.assert_array_equal(before["eligibility"],after["eligibility"])
    assert before["target_valid"].all()
    assert not after["target_valid"].any()


def test_future_volume_frozen_origin_baseline_and_masks(panel):
    p=normalized(panel);sampler=DateBatchSampler(p,batch_dates=2,stocks_per_date=4)
    b=sampler();norm=p.metadata["normalizer"]["stock"]
    assert not b["future_stock_obs"][...,4:].any()
    assert not b["future_market_obs"][...,1:].any()
    for i,t in enumerate(b["origins"]):
        chosen=b["symbol_indices"][i]
        expected=np.log(p.volume[t+1:t+6,chosen].T/p.volume[t-19:t+1,chosen].mean(axis=0)[:,None])
        expected=np.clip((expected-norm["median"][3])/norm["iqr"][3],-10,10)
        np.testing.assert_allclose(b["future_stock"][i,:,:,3],expected,rtol=1e-6,atol=1e-6)
        assert p.dates[t+5]<=np.datetime64("2018-12-31")


def test_one_day_target_is_left_padded_and_padding_invalid(panel):
    p=normalized(panel);b=DateBatchSampler(p,batch_dates=2,stocks_per_date=9,horizon=1)()
    assert not b["future_stock_obs"][:,:,:4].any()
    assert not b["future_market_obs"][:,:4].any()
    assert not b["eligibility"][:,6:].any()
    assert not b["stock_obs"][:,6:].any()
    assert not b["target_valid"][:,6:].any()


def test_sampler_resume_and_cache_roundtrip(panel,tmp_path):
    p=normalized(panel);sampler=DateBatchSampler(p,batch_dates=2,stocks_per_date=4)
    state=copy.deepcopy(sampler.state_dict());expected=sampler()
    sampler.load_state_dict(state);actual=sampler()
    for key in expected: np.testing.assert_array_equal(actual[key],expected[key])
    path=tmp_path/"panel.npz";save_panel(p,path);restored=load_panel(path)
    assert restored.metadata==p.metadata
    np.testing.assert_array_equal(restored.stock,p.stock)
    np.testing.assert_array_equal(restored.dates,p.dates)


def test_monthly_selection_uses_previous_prices():
    dates=pd.bdate_range("2020-01-01",periods=150).to_numpy(dtype="datetime64[D]")
    raw=np.ones((150,3))*10;volume=np.ones_like(raw)
    volume[:,1]=2;volume[:,2]=3
    membership=np.ones_like(raw,bool)
    expected=monthly_universe(dates,membership,raw,volume,max_stocks=1)
    first_may=np.flatnonzero(dates.astype("datetime64[M]")==np.datetime64("2020-05"))[0]
    changed=raw.copy();changed[first_may:,0]=1e10
    actual=monthly_universe(dates,membership,changed,volume,max_stocks=1)
    np.testing.assert_array_equal(actual[:first_may+1],expected[:first_may+1])
    assert actual[first_may,2]


def test_janus_manual_offense_defense():
    idx=np.array([-.02,-.01,.01,.02])
    returns=np.column_stack((idx,idx+.005))
    # Index is benchmark-neutral (1-1); shifted asset is positive on both sides.
    expected=70.710678*((.03+.01)/.03-(-.03+.01)/(-.03))
    np.testing.assert_allclose(janus_score(returns,idx,200),[0,expected])


def test_reactivity_correct_first_nonzero_volume():
    close=np.arange(40,dtype=float)[:,None]+100
    volume=np.ones_like(close)*1000;volume[:4]=0
    full=reactivity(close+2,close-2,close,volume)
    suffix=reactivity(close[4:]+2,close[4:]-2,close[4:],volume[4:])
    np.testing.assert_allclose(full[4:],suffix,equal_nan=True)
    assert np.isnan(full[:14]).all()


def test_historical_membership_reverse_change():
    dates=np.array(["2020-01-01","2020-02-01","2020-03-01"],dtype="datetime64[D]")
    events=[{"date":"2020-02-01","symbol":"NEW","removedTicker":"OLD"}]
    actual=reconstruct_membership(dates,["OLD","NEW"],["NEW"],events,"2020-03-01")
    np.testing.assert_array_equal(actual,[[True,False],[False,True],[False,True]])


def test_safe_fmp_client_never_exposes_key(tmp_path):
    class Response:
        status_code=403
        def json(self):return {"message":"secret-data"}
    class Session:
        def get(self,url,params,headers,timeout):
            assert "secret-data" not in url
            assert "secret-data" not in json.dumps(params)
            assert headers["apikey"]=="secret-data"
            return Response()
    client=FMPClient(tmp_path,api_key="secret-data",session=Session())
    with pytest.raises(DataAccessError) as captured:client.get("test")
    assert "secret-data" not in str(captured.value)
    assert "secret-data" not in repr(client)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changed",[{"batch_dates":3},{"stocks_per_date":3},
    {"start":"2016-06-01"},{"end":"2018-06-01"},{"min_stocks":3}])
def test_resume_rejects_changed_sampling_configuration(panel,changed):
    p=normalized(panel)
    state=DateBatchSampler(p,batch_dates=2,stocks_per_date=4).state_dict()
    config={"batch_dates":2,"stocks_per_date":4,**changed}
    with pytest.raises(ValueError,match="configuration changed"):
        DateBatchSampler(p,**config).load_state_dict(state)


def test_resume_rejects_normalizer_change(panel):
    p=normalized(panel)
    state=DateBatchSampler(p).state_dict()
    changed=copy.deepcopy(p)
    changed.metadata["normalizer"]["stock"]["median"][0]+=.01
    with pytest.raises(ValueError,match="normalizer_hash"):
        DateBatchSampler(changed).load_state_dict(state)


@pytest.mark.parametrize("symbol,column",[("STOCK","High"),("STOCK","Low"),
    ("XLB","Close"),("SPY","Open"),("SPY","Close"),("STOCK","RawClose"),("STOCK","RawVolume")])
def test_fingerprint_covers_every_feature_source(symbol,column):
    dates=pd.bdate_range("2010-01-01",periods=90)
    rng=np.random.default_rng(19);prices={}
    for name in ["STOCK","SPY",*REFERENCE_ETFS]:
        close=100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
        prices[name]=pd.DataFrame({"Open":close,"High":close*1.01,"Low":close*.99,
            "Close":close,"Volume":1e6,"RawClose":close,"RawVolume":1e6},index=dates)
    before=build_panel(prices,["STOCK"]).metadata["dataset_fingerprint"]
    prices[symbol].loc[dates[70],column]*=1.05
    after=build_panel(prices,["STOCK"]).metadata["dataset_fingerprint"]
    assert before!=after


def test_non_split_endpoint_uses_adj_prefixed_raw_fields():
    frame=_prices_frame([{"date":"2024-01-02","adjOpen":400.,"adjClose":402.,"volume":1234}],adjusted=False)
    assert frame.iloc[0]["RawOpen"]==400
    assert frame.iloc[0]["RawClose"]==402
    assert "Close" not in frame


@pytest.mark.parametrize("adjusted",[True,False])
def test_capped_history_backfills_only_missing_older_dates(adjusted):
    days=pd.bdate_range("2000-01-03",periods=5017)
    class Client:
        def __init__(self):self.calls=[]
        def get(self,endpoint,params):
            self.calls.append((endpoint,dict(params)))
            selected=days[(days>=pd.Timestamp(params["from"]))&(days<=pd.Timestamp(params["to"]))][-5000:]
            return [{"date":str(day.date()),"adjOpen":10.,"adjHigh":11.,"adjLow":9.,"adjClose":10.5,"volume":100.}
                    for day in selected[::-1]]
    client=Client()
    result=fetch_price_history(client,"SPY",str(days[0].date()),str(days[-1].date()),adjusted=adjusted)
    assert len(result)==5017
    assert len(client.calls)==2
    assert client.calls[1][1]["to"]==str((days[17]-pd.Timedelta(days=1)).date())
    assert result.index.is_monotonic_increasing and result.index.is_unique
    assert ("Open" if adjusted else "RawOpen") in result


def test_short_ipo_history_does_not_trigger_backfill():
    class Client:
        calls=0
        def get(self,endpoint,params):
            self.calls+=1
            return [{"date":day,"adjOpen":10.,"adjHigh":11.,"adjLow":9.,"adjClose":10.5,"volume":100.}
                    for day in ("2017-01-03","2017-01-04","2017-01-05")]
    client=Client()
    result=fetch_price_history(client,"IPO","2003-01-01","2025-12-31")
    assert len(result)==3 and client.calls==1


def test_capped_history_no_progress_fails_safely():
    days=pd.bdate_range("2005-01-03",periods=5000)
    payload=[{"date":str(day.date()),"adjOpen":10.,"adjHigh":11.,"adjLow":9.,"adjClose":10.5,"volume":100.}
             for day in days]
    class Client:
        calls=0
        def get(self,endpoint,params):self.calls+=1;return payload
    client=Client()
    with pytest.raises(DataAccessError,match="backfill_no_progress"):
        fetch_price_history(client,"SPY","2003-01-01","2025-12-31")
    assert client.calls==2
