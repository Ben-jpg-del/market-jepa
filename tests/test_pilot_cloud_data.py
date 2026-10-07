import json
import numpy as np
import pandas as pd
import pytest
from src import cloud_data as cd
from src.data import DataAccessError,load_panel
from src.features import REFERENCE_ETFS


def frame(start="2016-01-01",end="2018-12-31",seed=1):
    dates=pd.bdate_range(start,end);rng=np.random.default_rng(seed)
    close=100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
    volume=rng.lognormal(14,.2,len(dates))
    return pd.DataFrame({"Open":close,"High":close*1.01,"Low":close*.99,"Close":close,
        "Volume":volume,"RawOpen":close,"RawClose":close,"RawVolume":volume},index=dates)


def rows(df):
    return [{"date":str(t.date()),"adjOpen":r.Open,"adjHigh":r.High,"adjLow":r.Low,
             "adjClose":r.Close,"volume":r.Volume} for t,r in df.iterrows()]


class FakeClient:
    def __init__(self,df): self.df=df;self.calls=[]
    def get(self,endpoint,params=None,refresh=False):
        self.calls.append((endpoint,params,refresh))
        if endpoint=="historical-price-eod/dividend-adjusted": return rows(self.df)
        raise DataAccessError("http_error",endpoint,403)


def mock_preflight(monkeypatch):
    monkeypatch.setattr(cd,"preflight",lambda client:{"ok":True,"checks":{"raw_prices":{"ok":True}}})


def test_truncated_history_stops_before_broad_download(monkeypatch,tmp_path):
    mock_preflight(monkeypatch)
    client=FakeClient(frame("2019-01-01","2020-12-31"))
    monkeypatch.setattr(cd,"download_dataset",lambda *a,**kw:pytest.fail("Broad download must not run"))
    with pytest.raises(cd.PreparationError,match="truncated"):
        cd.prepare_cloud_data(tmp_path,client=client,progress=lambda x:None)
    report=json.loads((tmp_path/"coverage.json").read_text())
    assert report["status"]=="blocked"
    assert not report["benchmark_coverage"]["pre2019_history_present"]
    assert len(client.calls)==1


def test_historical_failure_requires_explicit_diagnostic(monkeypatch,tmp_path):
    mock_preflight(monkeypatch)
    client=FakeClient(frame())
    with pytest.raises(cd.PreparationError,match="requires --allow-diagnostic"):
        cd.prepare_cloud_data(tmp_path,start="2016-01-01",end="2018-12-31",client=client,progress=lambda x:None)
    report=json.loads((tmp_path/"coverage.json").read_text())
    assert report["historical_candidate_failure"]["status"]==403


def test_historical_reconstruction_uses_full_union_and_snapshot():
    class MembershipClient:
        def get(self,endpoint,params=None,refresh=False):
            assert refresh
            if endpoint=="sp500-constituent":return [{"symbol":f"S{i:03}"} for i in range(400)]
            return [{"date":"2000-01-01","symbol":"S000","removedTicker":"ANCIENT"},
                    {"date":"2020-01-01","symbol":"S001","removedTicker":"OLD"}]
    dates=np.array(["2016-01-01","2020-01-01"],dtype="datetime64[D]")
    symbols,membership,report=cd.historical_candidate(MembershipClient(),dates,"2026-01-01")
    assert "ANCIENT" not in symbols
    assert membership[0,symbols.index("OLD")]
    assert not membership[1,symbols.index("OLD")]
    assert not membership[0,symbols.index("S001")]
    assert membership[1,symbols.index("S001")]
    assert not report["point_in_time_verified"]


def test_preparation_prunes_after_selection_and_accounts_missing(monkeypatch,tmp_path):
    mock_preflight(monkeypatch);spy=frame();dates=spy.index.to_numpy(dtype="datetime64[D]")
    names=["A","B","C","MISSING"]
    membership=np.ones((len(dates),4),bool)
    monkeypatch.setattr(cd,"historical_candidate",lambda *a: (names,membership,{"mode":"historical_candidate"}))
    prices={s:frame(seed=i+2) for i,s in enumerate(["A","B","C","SPY",*REFERENCE_ETFS])}
    prices["SPY"]=spy
    # C never enters the top two. MISSING has no vendor history at all.
    prices["C"]["RawVolume"]=1.
    monkeypatch.setattr(cd,"download_dataset",lambda *a,**kw:(prices,{"adjusted_failures":{"MISSING":{"code":"empty_history"}}}))
    panel,report=cd.prepare_cloud_data(tmp_path,start="2016-01-01",end="2018-12-31",max_stocks=2,
        client=FakeClient(spy),progress=lambda x:None)
    assert panel.symbols==["A","B"]
    assert report["never_selected_symbols"]==["C","MISSING"]
    assert report["stock_coverage"]["MISSING"]["adjusted_rows"]==0
    assert report["stock_coverage"]["MISSING"]["member_dates"]==len(dates)
    assert report["status"]=="ready_research_only"
    assert not report["point_in_time_verified"]
    assert report["original_candidate_count"]==4
    assert panel.metadata["source_fingerprint"]==report["source_fingerprint"]
    loaded=load_panel(tmp_path/"panel_raw.npz")
    np.testing.assert_array_equal(loaded.stock_obs,panel.stock_obs)


def test_source_fingerprint_includes_unselected_stock():
    prices={"A":frame(),"NEVER_SELECTED":frame(seed=2)}
    membership=np.ones((len(prices["A"]),2),bool)
    before=cd._source_fingerprint(prices,list(prices),membership)
    prices["NEVER_SELECTED"].iloc[10,prices["NEVER_SELECTED"].columns.get_loc("RawVolume")]*=2
    assert cd._source_fingerprint(prices,list(prices),membership)!=before


def test_malformed_membership_row_fails_closed():
    class Client:
        def get(self,endpoint,params=None,refresh=False):
            if endpoint=="sp500-constituent":return [{"symbol":f"S{i:03}"} for i in range(400)]
            return [{"dateAdded":"2000-01-01","ticker":"XYZ"}]
    with pytest.raises(cd.PreparationError,match="schema"):
        cd.historical_candidate(Client(),np.array(["2016-01-01"],dtype="datetime64[D]"),"2026-01-01")
