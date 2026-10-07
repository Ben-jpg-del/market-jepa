import csv
import io
import numpy as np
import pandas as pd
import pytest
from src.expanded_data import (HoldingsSnapshot,HoldingsUnavailable,parse_iwb_csv,IWB,IWM,
    IWBHistoryClient,monthly_request_dates,fetch_monthly_history,membership_from_snapshots,
    save_roster_bundle,load_roster_bundle)


def csv_body(date="31/Jan/2018",tickers=("AAPL","BRK B")):
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(["iShares Russell 1000 ETF"])
    writer.writerow(["Fund Holdings as of",date])
    writer.writerow(["Ticker","Name","Sector","Asset Class","Market Value","Weight (%)"])
    for ticker in tickers:writer.writerow([ticker,ticker+" name","Technology","Equity","1’000.00","0.5"])
    writer.writerow(["USD","Cash","Cash","Cash","500.00","0.1"])
    writer.writerow(["Legal footer text"])
    return ("\ufeff"+stream.getvalue()).encode()


def snapshot(date,symbols):
    return HoldingsSnapshot(date,date,"https://www.ishares.com/fixture","hash","2026-09-17T00:00:00Z",
        [{"symbol":s,"source_ticker":s} for s in symbols],[],0)


def test_parser_verifies_date_and_preserves_unmapped_symbols():
    result=parse_iwb_csv(csv_body(),"2018-01-31",min_equities=2)
    assert result.as_of_date=="2018-01-31"
    assert result.symbols==["AAPL"]
    assert result.unsupported_tickers==["BRK B"]
    assert result.holdings[0]["market_value"]==1000
    assert result.excluded_non_equity_rows==1
    mapped=parse_iwb_csv(csv_body(),"2018-01-31",symbol_map={"BRK B":"BRK-B"},min_equities=2)
    assert mapped.symbols==["AAPL","BRK-B"]


def test_ignored_historical_date_is_rejected():
    with pytest.raises(HoldingsUnavailable,match="as_of_date_mismatch"):
        parse_iwb_csv(csv_body("16/Sep/2026"),"2018-01-31",min_equities=2)
    with pytest.raises(HoldingsUnavailable,match="html_instead_of_csv"):
        parse_iwb_csv("<!DOCTYPE html><html>marketing page</html>","2018-01-31")


def test_unavailable_history_and_implausible_counts_rejected():
    with pytest.raises(HoldingsUnavailable,match="historical_date_unavailable"):
        parse_iwb_csv(csv_body("-"),"2006-01-31")
    with pytest.raises(HoldingsUnavailable,match="implausible_equity_count"):
        parse_iwb_csv(csv_body(),"2018-01-31")


def test_duplicate_mapping_cannot_merge_distinct_holdings():
    result=parse_iwb_csv(csv_body(tickers=("ABC","ABC A")),"2018-01-31",symbol_map={"ABC A":"ABC"},min_equities=2)
    assert result.symbols==[]
    assert all(row["exclusion_reason"]=="ambiguous_duplicate_source_identity" for row in result.holdings)
    assert result.duplicate_resolution[0]["resolution"]=="excluded_ambiguous_symbol_for_snapshot"
    assert len(result.duplicate_resolution[0]["original_source_rows"])==2


def duplicate_csv(prices=("10.00","10.00"),names=("SAME COMPANY","SAME COMPANY")):
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(["iShares Russell 1000 ETF"])
    writer.writerow(["Fund Holdings as of","31/Jan/2018"])
    writer.writerow(["Ticker","Name","Asset Class","Market Value","Weight (%)","Price","Exchange"])
    for price,name in zip(prices,names):writer.writerow(["ABC",name,"Equity","1000","0.5",price,"NYSE"])
    return stream.getvalue()


def test_identical_duplicate_rows_aggregate_and_preserve_originals():
    result=parse_iwb_csv(duplicate_csv(),"2018-01-31",min_equities=2)
    assert result.symbols==["ABC"] and len(result.holdings)==1
    assert result.holdings[0]["market_value"]==2000
    assert result.holdings[0]["weight_percent"]==1
    assert result.holdings[0]["aggregated_source_row_count"]==2
    assert result.holdings[0]["source_price"]==10
    assert len(result.duplicate_resolution[0]["original_source_rows"])==2


def test_same_name_different_prices_are_not_assumed_same_security():
    result=parse_iwb_csv(duplicate_csv(prices=("41.90","22.09")),"2018-01-31",min_equities=2)
    assert result.symbols==[]
    assert result.duplicate_resolution[0]["resolution"]=="excluded_ambiguous_symbol_for_snapshot"


def test_same_price_different_names_are_still_ambiguous():
    result=parse_iwb_csv(duplicate_csv(names=("ORIGINAL COMPANY","OTHER COMPANY")),"2018-01-31",min_equities=2)
    assert result.symbols==[]


def test_month_end_uses_supplied_trading_calendar():
    dates=np.array(["2018-03-28","2018-03-29","2018-04-02","2018-04-30"],dtype="datetime64[D]")
    assert monthly_request_dates(dates,"2018-03-01","2018-04-30")==["2018-03-29","2018-04-30"]


def test_history_allows_missing_prefix_but_fails_internal_gaps():
    dates=pd.bdate_range("2006-01-01","2006-04-30").to_numpy(dtype="datetime64[D]")
    class Client:
        def fetch(self,date):
            if date in ("2006-01-31","2006-03-31"):
                raise HoldingsUnavailable("historical_date_unavailable",date)
            return snapshot(date,["A","B"])
    with pytest.raises(HoldingsUnavailable,match="missing_internal_months") as error:
        fetch_monthly_history(Client(),dates,"2006-01-01","2006-04-30")
    assert error.value.coverage_report["missing_months"][0]["position"]=="prefix"
    snapshots,report=fetch_monthly_history(Client(),dates,"2006-01-01","2006-04-30",allow_internal_gaps=True)
    assert report["first_snapshot"]=="2006-02-28"
    assert report["valid_months"]==2
    assert not report["exact_russell_index_membership"]
    assert not report["point_in_time_verified"]


def test_membership_is_past_only_and_effective_next_session():
    dates=np.array(["2018-01-30","2018-01-31","2018-02-01","2018-02-28","2018-03-01"],dtype="datetime64[D]")
    snapshots=[snapshot("2018-01-31",["OLD","BOTH"]),snapshot("2018-02-28",["NEW","BOTH"])]
    symbols,member,report=membership_from_snapshots(dates,snapshots)
    assert not member[:2].any()
    assert member[2,symbols.index("OLD")] and not member[2,symbols.index("NEW")]
    assert member[3,symbols.index("OLD")] and not member[3,symbols.index("NEW")]
    assert member[4,symbols.index("NEW")] and not member[4,symbols.index("OLD")]
    assert report["snapshot_date_by_origin"][2]=="2018-01-31"
    assert report["first_covered_date"]=="2018-02-01"


def test_future_snapshot_does_not_change_past_membership():
    dates=pd.bdate_range("2018-01-01","2018-04-01").to_numpy(dtype="datetime64[D]")
    initial=[snapshot("2018-01-31",["A","B"])]
    before,first,_=membership_from_snapshots(dates,initial)
    after,second,_=membership_from_snapshots(dates,initial+[snapshot("2018-02-28",["B","C"])])
    rows=dates<=np.datetime64("2018-02-28")
    for name in before:np.testing.assert_array_equal(first[rows,before.index(name)],second[rows,after.index(name)])
    assert not second[rows,after.index("C")].any()


def test_stale_membership_is_never_carried_indefinitely():
    dates=np.array(["2018-02-01","2018-03-19"],dtype="datetime64[D]")
    _,member,report=membership_from_snapshots(dates,[snapshot("2018-01-31",["A"])],max_staleness_days=45)
    assert member[0,0] and not member[1,0]
    assert report["uncovered_dates"]==1


def test_bundle_roundtrip_preserves_source_provenance(tmp_path):
    snapshots=[snapshot("2018-01-31",["A","B"])];report={"point_in_time_verified":False}
    save_roster_bundle(snapshots,report,tmp_path/"roster.json")
    restored,restored_report=load_roster_bundle(tmp_path/"roster.json")
    assert restored==snapshots and restored_report==report


def test_duplicate_resolution_survives_bundle_roundtrip(tmp_path):
    value=parse_iwb_csv(duplicate_csv(prices=("41.90","22.09")),"2018-01-31",min_equities=2)
    save_roster_bundle([value],{},tmp_path/"roster.json")
    restored,_=load_roster_bundle(tmp_path/"roster.json")
    assert restored[0].duplicate_resolution==value.duplicate_resolution
    assert restored[0].symbols==[]
    assert restored[0].holdings[1]["source_price"]==22.09


def _as_iwm(body):
    return body.replace(b"iShares Russell 1000 ETF",b"iShares Russell 2000 ETF") if isinstance(body,bytes) else body.replace("iShares Russell 1000 ETF","iShares Russell 2000 ETF")


def test_fund_spec_selects_title_bounds_and_source_url():
    body=_as_iwm(csv_body())
    with pytest.raises(HoldingsUnavailable) as err:parse_iwb_csv(body,"2018-01-31",min_equities=2)
    assert err.value.code=="wrong_fund"
    result=parse_iwb_csv(body,"2018-01-31",min_equities=2,spec=IWM)
    assert result.source_url==IWM.csv_url and "239710" in result.source_url
    with pytest.raises(HoldingsUnavailable) as err:parse_iwb_csv(body,"2018-01-31",spec=IWM)
    assert err.value.code=="implausible_equity_count"  # IWM bounds 1500-2300 reject a 2-row file
    assert IWB.min_equities==700 and IWM.max_equities==2300 and IWM.file_name=="IWM_holdings"


def test_client_cache_name_and_params_follow_the_fund(tmp_path):
    client=IWBHistoryClient(tmp_path,spec=IWM)
    assert client.spec is IWM
    body=_as_iwm(csv_body())
    (tmp_path/"IWM_2018-01-31.csv").write_bytes(body if isinstance(body,bytes) else body.encode())
    with pytest.raises(HoldingsUnavailable) as err:client.fetch("2018-01-31")
    assert err.value.code=="implausible_equity_count" and "IWM" in str(err.value)

