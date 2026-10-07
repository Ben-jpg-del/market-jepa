"""Apply the audited retrospective identity quarantine before research training.

This is a research exclusion, not a point-in-time identity certification. It
does not infer aliases or replace removed securities. The optional original arm
is written as a NEW copy; the archived original panel/report stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import MarketPanel, load_panel, save_panel
from src.features import STOCK_FEATURES


STOCK_AXIS_FIELDS = ("stock", "stock_obs", "open", "close", "volume", "volume_baseline20", "universe")
POLICY_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def load_policy(audit_path, expected_count=14, end_date="2021-12-31"):
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    conflicts = audit.get("priority_identity_conflicts")
    if not isinstance(conflicts, dict) or not conflicts or any(not isinstance(k,str) or not k for k in conflicts):
        raise ValueError("Audit needs a nonempty priority_identity_conflicts mapping")
    symbols = sorted(conflicts)
    if expected_count is not None and len(symbols) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} audited quarantine symbols; found {len(symbols)}")
    policy = {"version": POLICY_VERSION, "requested_symbols": symbols,
              "audit_sha256": file_sha256(audit_path), "audit_path": str(audit_path),
              "end_date": end_date, "scope": "all retained history, no replacement entrants",
              "identity_conflicts": conflicts,
              "reason": "Conservative retrospective research exclusion of unresolved security-identity conflicts; not a point-in-time tradable rule or verified identifier mapping",
              "janus_recomputation": "Average-rank remaining finite, origin-eligible fractiles, rescale to [0,1], then recompute five-session differences; fixed reference gate and per-stock score order are unchanged"}
    identity = {key: value for key,value in policy.items() if key != "audit_path"}
    policy["policy_fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return policy


def rerank_janus(fractiles, observed, universe):
    """Exact rank-order restriction for the fixed-reference JANUS extension.

    Existing fractiles are monotonic in unchanged per-stock scores. Average
    ranks preserve ties; a remaining cohort with fewer than two valid scores
    remains unavailable. No future eligibility or observations are consulted.
    """
    fractiles = np.asarray(fractiles)
    if fractiles.shape != observed.shape or fractiles.shape != universe.shape:
        raise ValueError("JANUS values, validity, and universe must have identical [date,stock] shapes")
    ranks = np.full(fractiles.shape, np.nan, dtype=np.float64)
    for t in range(len(fractiles)):
        valid = np.asarray(observed[t],bool) & np.asarray(universe[t],bool) & np.isfinite(fractiles[t])
        count = int(valid.sum())
        if count >= 2:
            ranks[t,valid] = (rankdata(fractiles[t,valid],method="average")-1)/(count-1)
    delta = np.full_like(ranks,np.nan)
    delta[5:] = ranks[5:]-ranks[:-5]
    return ranks.astype(np.float32), delta.astype(np.float32)


def _derived_fingerprint(panel, parent_fingerprint, policy):
    digest = hashlib.sha256(json.dumps({"parent_dataset_fingerprint":parent_fingerprint,
        "policy":policy,"symbols":panel.symbols},sort_keys=True).encode())
    for name,value in sorted(vars(panel).items()):
        if isinstance(value,np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(json.dumps({"name":name,"shape":array.shape,"dtype":str(array.dtype)},sort_keys=True).encode())
            buffer = memoryview(array.view(np.uint8)).cast("B")
            for first in range(0,len(buffer),1024*1024):
                digest.update(buffer[first:first+1024*1024])
    return digest.hexdigest()


def read_origin_availability(path):
    """Read only small origin fields, without duplicating the expanded features."""
    with np.load(path,allow_pickle=False) as payload:
        return {"dates":payload["dates"].copy(),"symbols":payload["symbols"].tolist(),
                "universe":payload["universe"].copy(),
                "dataset_fingerprint":json.loads(str(payload["metadata"]))["dataset_fingerprint"]}


def quarantine_panel(panel, policy, parent_panel_sha256=None, origin_availability=None):
    """Pure transformation: drop every stock axis, recompute ranks, retain proof."""
    if panel.metadata.get("normalizer"):
        raise ValueError("Quarantine requires a raw panel; normalized/clipped ranks cannot be reconstructed safely")
    previous = panel.metadata.get("identity_quarantine")
    if previous:
        if previous.get("policy_fingerprint") != policy["policy_fingerprint"]:
            raise ValueError("Panel already has a different identity quarantine policy")
        return panel
    if panel.stock.shape[-1] != 14:
        raise ValueError("Expected the 14-channel book feature schema")
    names = panel.metadata.get("stock_features")
    if names is not None and list(names) != list(STOCK_FEATURES):
        raise ValueError("Stock channel definitions differ from the fixed pilot schema")
    if len(set(panel.symbols)) != len(panel.symbols):
        raise ValueError("Panel contains duplicate stock symbols")
    old_dates,old_count = len(panel.dates),len(panel.symbols)
    excluded = set(policy["requested_symbols"])
    keep = np.asarray([symbol not in excluded for symbol in panel.symbols],bool)
    if not keep.any():
        raise ValueError("The quarantine would remove every stock")
    end = policy.get("end_date")
    count = old_dates if end is None else int(np.searchsorted(panel.dates,np.datetime64(end,"D"),side="right"))
    if count == 0:
        raise ValueError("No dates remain after the requested history trim")
    values = {}
    indices = np.flatnonzero(keep)
    for name,value in vars(panel).items():
        if isinstance(value,np.ndarray):
            if not value.ndim or len(value) != old_dates:
                raise ValueError(f"Unexpected panel array date axis: {name}")
            if name in STOCK_AXIS_FIELDS:
                if value.ndim < 2 or value.shape[1] != old_count:
                    raise ValueError(f"Unexpected stock axis: {name}")
                values[name] = np.ascontiguousarray(np.take(value[:count],indices,axis=1))
            else:
                values[name] = np.ascontiguousarray(value[:count])
    values["symbols"] = [symbol for symbol,kept in zip(panel.symbols,keep) if kept]
    if origin_availability is not None:
        reference=origin_availability
        if policy.get("origin_availability_dataset_fingerprint")!=reference["dataset_fingerprint"]:
            raise ValueError("Origin availability does not match the declared policy")
        ref_dates=np.asarray(reference["dates"])
        positions=np.searchsorted(ref_dates,values["dates"])
        valid=positions<len(ref_dates)
        valid[valid]&=ref_dates[positions[valid]]==values["dates"][valid]
        ref_symbols={symbol:k for k,symbol in enumerate(reference["symbols"])}
        allowed=np.zeros_like(values["universe"],dtype=bool)
        for k,symbol in enumerate(values["symbols"]):
            if symbol in ref_symbols:
                allowed[valid,k]=reference["universe"][positions[valid],ref_symbols[symbol]]
        values["universe"] &= allowed
    ranks,delta = rerank_janus(values["stock"][...,12],values["stock_obs"][...,12],values["universe"])
    values["stock"][...,12],values["stock"][...,13] = ranks,delta
    values["stock_obs"][...,12],values["stock_obs"][...,13] = np.isfinite(ranks),np.isfinite(delta)
    provenance = {**copy.deepcopy(policy), "parent_dataset_fingerprint":panel.metadata["dataset_fingerprint"],
        "parent_panel_sha256":parent_panel_sha256,"parent_stock_count":old_count,
        "parent_date_count":old_dates,"retained_stock_count":int(keep.sum()),"retained_date_count":count,
        "removed_symbols":[s for s in panel.symbols if s in excluded],
        "already_absent_symbols":sorted(excluded-set(panel.symbols)),
        "recomputed_features":[STOCK_FEATURES[12],STOCK_FEATURES[13]],
        "replacement_symbols_added":False,"point_in_time_verified":False}
    metadata = copy.deepcopy(panel.metadata)
    metadata.update(identity_quarantine=provenance,normalizer=None,point_in_time_verified=False)
    if origin_availability is not None:
        metadata["universe_label"]="Original up-to-128 selected pool filtered by expanded origin availability; no replacements"
        covered=np.flatnonzero(np.asarray(origin_availability["universe"]).any(axis=1))
        if len(covered):
            metadata["first_membership_date"]=str(origin_availability["dates"][covered[0]])
    metadata.setdefault("limitations",[]).append(policy["reason"])
    metadata.setdefault("feature_notes",{})["identity_quarantine"] = policy["janus_recomputation"]
    values["metadata"] = metadata
    result = MarketPanel(**values)
    result.metadata["dataset_fingerprint"] = _derived_fingerprint(result,panel.metadata["dataset_fingerprint"],policy)
    return result


def panel_counts(panel, first_covered_date=None):
    count = np.asarray(panel.universe,bool).sum(axis=1)
    start = max(np.datetime64("2006-01-01"), np.datetime64(first_covered_date)) if first_covered_date else np.datetime64("2006-01-01")
    train = (panel.dates>=start)&(panel.dates<=np.datetime64("2018-12-31"))
    selection = {"train_dates":int(train.sum()),"minimum_train_count":int(count[train].min()) if train.any() else 0,
        "median_train_count":float(np.median(count[train])) if train.any() else None,
        "maximum_count":int(count.max()),"train_dates_under_128":int((count[train]<128).sum()),
        "train_dates_under_32":int((count[train]<32).sum()),"total_selected_stock_dates":int(count.sum())}
    # Rolling completeness is computed without a full floating-point panel.
    fast = panel.stock_obs[...,:4].all(axis=-1)
    running = np.zeros(len(panel.symbols),np.int32)
    eligible_count = np.zeros(len(panel.dates),np.int32)
    for t in range(len(panel.dates)):
        running += fast[t]
        if t>=60:running -= fast[t-60]
        if t>=59:eligible_count[t] = np.count_nonzero(panel.universe[t]&(running==60))
    times=np.arange(len(panel.dates))
    cutoff=np.searchsorted(panel.dates,np.datetime64("2018-12-31"),side="right")-1
    samplers={}
    for horizon in (1,5):
        valid=(times>=59)&(times+horizon<=cutoff)&(times+horizon<len(times))
        valid&=(panel.dates>=np.datetime64("2006-01-01"))&(eligible_count>=2)&panel.market_obs[:,0]
        samplers[str(horizon)]={"origins":int(valid.sum()),"eligible_contexts":int(eligible_count[valid].sum())}
    return selection,samplers


def updated_coverage(panel,parent_coverage,panel_path,panel_sha256):
    report = copy.deepcopy(parent_coverage)
    provenance=copy.deepcopy(panel.metadata["identity_quarantine"])
    first = report.get("timing",{}).get("first_covered_date") or panel.metadata.get("first_membership_date")
    selection,samplers=panel_counts(panel,first)
    original_stock_coverage=copy.deepcopy(report.get("stock_coverage",{}))
    retained = {}
    for k,symbol in enumerate(panel.symbols):
        item=copy.deepcopy(original_stock_coverage.get(symbol,{}))
        if item:
            item["parent_download_coverage_before_trim"] = {key:item[key] for key in ("adjusted_rows","raw_rows","member_dates","selected_dates") if key in item}
        item.update(adjusted_rows=int(np.isfinite(panel.close[:,k]).sum()),
                    selected_dates=int(panel.universe[:,k].sum()),quarantined=False)
        # Raw source data and membership are not retained in MarketPanel. Their
        # old counts must not masquerade as recomputed post-trim statistics.
        item.pop("raw_rows",None);item.pop("member_dates",None)
        retained[symbol]=item
    excluded = {symbol:{**original_stock_coverage.get(symbol,{}),"selected_dates_after_quarantine":0,
                        "quarantined":True} for symbol in provenance["requested_symbols"]}
    report.update(status="ready_research_only",identity_quarantine=provenance,
        dataset_fingerprint=panel.metadata["dataset_fingerprint"],point_in_time_verified=False,
        start=str(panel.dates[0]),end=str(panel.dates[-1]),stock_shape=list(panel.stock.shape),
        market_shape=list(panel.market.shape),retained_symbols=list(panel.symbols),retained_symbol_count=len(panel.symbols),
        selection=selection,samplers=samplers,stock_coverage=retained,quarantined_stock_coverage=excluded,
        panel_path=str(panel_path),panel_sha256=panel_sha256,
        count_scope_note="Selection, adjusted-row and sampler counts recomputed on retained dates/stocks. Raw-row and membership-source counts retained only under parent_download_coverage_before_trim.")
    report["limitations"] = list(dict.fromkeys([*report.get("limitations",[]),provenance["reason"]]))
    if "universe_label" in panel.metadata:
        report["universe_label"]=panel.metadata["universe_label"]
    if "benchmark_coverage" in report:
        report["parent_benchmark_coverage_before_trim"]=report.pop("benchmark_coverage")
    benchmark_valid=np.isfinite(panel.benchmark_close)
    report["benchmark_coverage"]={"rows":len(panel.dates),"usable_rows":int(benchmark_valid.sum()),
        "first_date":str(panel.dates[benchmark_valid][0]) if benchmark_valid.any() else None,
        "last_date":str(panel.dates[benchmark_valid][-1]) if benchmark_valid.any() else None,
        "count_scope":"Retained panel dates; original source download checks retained separately"}
    if "never_selected_symbols" in report:
        report["never_selected_symbols"] = sorted(set(report["never_selected_symbols"])-set(provenance["requested_symbols"]))
    return report


def quarantine_files(source_panel_path,source_coverage_path,audit_path,*,output_panel_path=None,
                     output_coverage_path=None,expected_count=14,end_date="2021-12-31",
                     origin_availability_path=None):
    source_panel_path,source_coverage_path=Path(source_panel_path),Path(source_coverage_path)
    destination=Path(output_panel_path) if output_panel_path else source_panel_path
    coverage_destination=Path(output_coverage_path) if output_coverage_path else source_coverage_path
    destination.parent.mkdir(parents=True,exist_ok=True)
    coverage_destination.parent.mkdir(parents=True,exist_ok=True)
    policy=load_policy(audit_path,expected_count,end_date)
    availability=read_origin_availability(origin_availability_path) if origin_availability_path else None
    if availability is not None:
        policy["origin_availability_dataset_fingerprint"]=availability["dataset_fingerprint"]
        policy["origin_availability_rule"]="Intersect original selected pool with expanded origin availability by exact date and symbol before JANUS ranking; no replacements"
        policy["policy_fingerprint"]=hashlib.sha256(json.dumps({k:v for k,v in policy.items()
            if k not in ("audit_path","policy_fingerprint")},sort_keys=True).encode()).hexdigest()
    coverage=json.loads(source_coverage_path.read_text())
    if coverage.get("status") != "ready_research_only":
        raise RuntimeError("Source data preparation is not complete; quarantine was not applied")
    source_sha=file_sha256(source_panel_path)
    candidate=load_panel(destination) if destination.exists() else None
    applied=candidate.metadata.get("identity_quarantine") if candidate is not None else None
    if applied:
        if applied.get("policy_fingerprint") != policy["policy_fingerprint"]:
            raise ValueError("Destination already contains a different quarantine policy")
        if source_panel_path.resolve()!=destination.resolve() and applied.get("parent_panel_sha256")!=source_sha:
            raise ValueError("Existing quarantined copy belongs to a different source panel")
        # Also completes a prior interruption between the two atomic renames.
        panel=candidate
    else:
        if coverage.get("panel_sha256") and coverage["panel_sha256"]!=source_sha:
            raise ValueError("Source panel SHA differs from its completed coverage report")
        original=candidate if source_panel_path.resolve()==destination.resolve() else load_panel(source_panel_path)
        panel=quarantine_panel(original,policy,source_sha,availability)
        del original,candidate
    backup_panel=destination.with_name(destination.stem+".pre_identity_quarantine"+destination.suffix)
    backup_coverage=coverage_destination.with_name(coverage_destination.stem+".pre_identity_quarantine"+coverage_destination.suffix)
    if not applied:
        if not backup_panel.exists():
            try:os.link(source_panel_path,backup_panel)
            except OSError:shutil.copy2(source_panel_path,backup_panel)
        elif file_sha256(backup_panel)!=source_sha:
            raise ValueError("Pre-quarantine backup belongs to a different source panel")
        if not backup_coverage.exists():
            write_json_atomic(backup_coverage,coverage)
    parent_coverage=json.loads(backup_coverage.read_text()) if backup_coverage.exists() else coverage
    temporary=None
    try:
        if not applied:
            with tempfile.NamedTemporaryFile(dir=destination.parent,suffix=".npz",delete=False) as handle:
                temporary=Path(handle.name)
            save_panel(panel,temporary)
            with temporary.open("rb") as handle:os.fsync(handle.fileno())
            new_sha=file_sha256(temporary)
        else:new_sha=file_sha256(destination)
        report=updated_coverage(panel,parent_coverage,destination,new_sha)
        report["identity_quarantine"]["parent_panel_backup"]=str(backup_panel)
        report["identity_quarantine"]["parent_coverage_backup"]=str(backup_coverage)
        coverage_temp=coverage_destination.with_suffix(coverage_destination.suffix+".quarantine.tmp")
        coverage_temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if temporary is not None:
            os.replace(temporary,destination)
            temporary=None
        os.replace(coverage_temp,coverage_destination)
        write_json_atomic(destination.with_suffix(".identity_quarantine.json"),report["identity_quarantine"])
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return {"panel":str(destination),"coverage":str(coverage_destination),"panel_sha256":new_sha,
            "dataset_fingerprint":panel.metadata["dataset_fingerprint"],
            "removed_symbols":panel.metadata["identity_quarantine"]["removed_symbols"],
            "already_absent_symbols":panel.metadata["identity_quarantine"]["already_absent_symbols"],
            "remaining_symbols":len(panel.symbols),"selection":report["selection"],
            "idempotent_existing_quarantine":bool(applied)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,default=ROOT/"work/fjepa-expanded-data/panel_raw.npz")
    parser.add_argument("--coverage",type=Path,default=ROOT/"work/fjepa-expanded-data/coverage.json")
    parser.add_argument("--audit",type=Path,default=ROOT/"outputs/universe-expansion/holdings_identity_audit.json")
    parser.add_argument("--end-date",default="2021-12-31")
    parser.add_argument("--also-original",action="store_true")
    parser.add_argument("--original-panel",type=Path,default=ROOT/"work/fjepa-cloud-data/panel_raw.npz")
    parser.add_argument("--original-coverage",type=Path,default=ROOT/"work/fjepa-cloud-data/coverage.json")
    args=parser.parse_args(argv)
    result={"expanded":quarantine_files(args.panel,args.coverage,args.audit,end_date=args.end_date)}
    if args.also_original:
        result["original"]=quarantine_files(args.original_panel,args.original_coverage,args.audit,
            output_panel_path=args.panel.parent/"original_panel.npz",
            output_coverage_path=args.panel.parent/"original_coverage.json",end_date=args.end_date,
            origin_availability_path=args.panel)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
