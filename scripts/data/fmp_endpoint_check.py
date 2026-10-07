"""Probe FMP endpoints relevant to new information families: access, history depth, timestamps.

Uses the project's FMPClient (stable API, header auth, cached responses under
work/fmp-live-cache). The key comes from FMP_API_KEY in the environment and is never printed.
For each endpoint: HTTP outcome, row count, which date-like fields exist, earliest/latest date,
and a few sample keys. Writes outputs/data-sources/fmp_endpoint_check.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DataAccessError, FMPClient  # noqa: E402

OUT = ROOT / "outputs" / "data-sources"
DATE_FIELDS = ("date", "fillingDate", "filingDate", "acceptedDate", "publishedDate", "reportedDate",
               "announcementDate", "transactionDate", "filedDate", "fiscalDateEnding", "paymentDate", "recordDate")

PROBES = [
    # family 1: earnings events
    ("earnings", "earnings", {"symbol": "AAPL", "limit": 400}),
    ("earnings_surprises", "earnings-surprises", {"symbol": "AAPL"}),
    ("earnings_calendar_2010", "earnings-calendar", {"from": "2010-01-01", "to": "2010-03-31"}),
    # family 2: point-in-time fundamentals
    ("income_statement_q", "income-statement", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("balance_sheet_q", "balance-sheet-statement", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("cash_flow_q", "cash-flow-statement", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("key_metrics_q", "key-metrics", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("ratios_q", "ratios", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("income_statement_q_delisted_example", "income-statement", {"symbol": "GE", "period": "quarter", "limit": 400}),
    # family 3: analysts
    ("analyst_estimates_q", "analyst-estimates", {"symbol": "AAPL", "period": "quarter", "limit": 400}),
    ("grades_historical", "grades-historical", {"symbol": "AAPL", "limit": 400}),
    ("price_target_summary", "price-target-summary", {"symbol": "AAPL"}),
    # family 4: macro
    ("economic_calendar_2010", "economic-calendar", {"from": "2010-01-01", "to": "2010-03-31"}),
    ("economic_calendar_2006", "economic-calendar", {"from": "2006-01-01", "to": "2006-03-31"}),
    ("treasury_rates_2006", "treasury-rates", {"from": "2006-01-01", "to": "2006-03-31"}),
    ("economic_indicator_cpi", "economic-indicators", {"name": "CPI", "from": "2006-01-01", "to": "2006-12-31"}),
    # family 5: text / filings / insiders
    ("stock_news_2012", "news/stock", {"symbols": "AAPL", "from": "2012-01-01", "to": "2012-03-31", "limit": 250}),
    ("stock_news_2020", "news/stock", {"symbols": "AAPL", "from": "2020-01-01", "to": "2020-03-31", "limit": 250}),
    ("press_releases", "news/press-releases", {"symbols": "AAPL", "limit": 250}),
    ("sec_filings_2008", "sec-filings-search/symbol", {"symbol": "AAPL", "from": "2008-01-01", "to": "2008-12-31"}),
    ("insider_trading", "insider-trading/search", {"symbol": "AAPL", "limit": 400}),
    # corporate actions / universe
    ("dividends", "dividends", {"symbol": "AAPL"}),
    ("splits", "splits", {"symbol": "AAPL"}),
    ("delisted_companies", "delisted-companies", {"limit": 500}),
]


def summarize(rows):
    if not rows:
        return {"rows": 0}
    keys = sorted({k for r in rows[:50] for k in r})
    present = [f for f in DATE_FIELDS if f in rows[0]]
    dates = {}
    for f in present:
        values = sorted(str(r[f])[:10] for r in rows if r.get(f))
        if values:
            dates[f] = {"earliest": values[0], "latest": values[-1]}
    return {"rows": len(rows), "date_fields": present, "dates": dates, "sample_keys": keys[:25]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = FMPClient(ROOT / "work" / "fmp-live-cache")
    results = {}
    for label, endpoint, params in PROBES:
        try:
            rows = client.get(endpoint, params)
            results[label] = {"endpoint": endpoint, "params": params, "ok": True, **summarize(rows)}
        except DataAccessError as exc:
            results[label] = {"endpoint": endpoint, "params": params, "ok": False, **exc.as_dict()}
        r = results[label]
        print(json.dumps({"probe": label, "ok": r["ok"], "rows": r.get("rows"), "dates": r.get("dates"),
                          "error": None if r["ok"] else {k: v for k, v in r.items() if k in ("code", "status")}}), flush=True)
    (OUT / "fmp_endpoint_check.json").write_text(json.dumps(results, indent=1))
    lines = ["# FMP endpoint check (2026-09-19)", "", "| probe | endpoint | ok | rows | date fields | earliest | latest |", "|---|---|---|---:|---|---|---|"]
    for label, r in results.items():
        if r["ok"]:
            d = r.get("dates", {})
            first = min((v["earliest"] for v in d.values()), default="")
            last = max((v["latest"] for v in d.values()), default="")
            lines.append(f"| {label} | `{r['endpoint']}` | yes | {r['rows']} | {', '.join(r.get('date_fields', []))} | {first} | {last} |")
        else:
            lines.append(f"| {label} | `{r['endpoint']}` | **no** ({r.get('code')} {r.get('status') or ''}) | | | | |")
    (OUT / "fmp_endpoint_check.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "fmp_endpoint_check.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
