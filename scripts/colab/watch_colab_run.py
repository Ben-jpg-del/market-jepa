#!/usr/bin/env python3
"""Local lifecycle watcher for the explicitly authorized four-hour Colab run.

This uses only the official CLI's exec/download/stop commands. It never reads
authentication files or prints CLI output (which may contain sensitive errors).
Run as a detached process only after the cloud job has started. Laptop sleep,
shutdown, revoked access or loss of network can prevent timely runtime release;
the cloud job must keep its own independent process deadline.

Example (root operator supplies the actual UTC job start):
  python scripts/colab/watch_colab_run.py --job-start 2026-09-17T12:00:00+00:00

No scheduler, cron entry, task creation or process launch occurs on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import tarfile
import time


ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CLI=ROOT/"work/colab-client-env/bin/colab"
DEFAULT_OUTPUT=ROOT/"outputs/colab/live-run"
MARKER="FJEPA_WATCH_STATE:"
TERMINAL={"completed","completed_with_deferred_intermediate","failed","paused_deadline","paused_interrupt"}


def utc_now(): return datetime.now(timezone.utc).isoformat()


def parse_start(value):
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None: raise ValueError("job-start requires an explicit timezone")
    return parsed.timestamp()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)


def safe_relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe remote relative path")
    return path


def remote_probe_code(remote_output,state_file="job_state.json"):
    """Read only the authorized run directory; no kernel state/auth inspection."""
    if PurePosixPath(state_file).name != state_file or not state_file.endswith('.json'):
        raise ValueError("State file must be a JSON basename")
    return '''import json, hashlib
from pathlib import Path
_watch_root=Path(ROOT_LITERAL).resolve()
_watch_state_path=_watch_root/STATE_LITERAL
_watch_state={}
if _watch_state_path.is_file():
    try:
        _watch_state=json.loads(_watch_state_path.read_text())
    except (OSError,json.JSONDecodeError):
        pass
_watch_files=[]
_watch_latest={}
if _watch_root.is_dir():
    for _watch_file in _watch_root.rglob('*'):
        if not _watch_file.is_file() or _watch_file.is_symlink():
            continue
        _watch_rel=_watch_file.relative_to(_watch_root)
        if 'expanded_evaluation_cache' in _watch_rel.parts:
            continue
        if _watch_file.suffix=='.pt' and _watch_file.name.startswith('step_'):
            _watch_key=str(_watch_rel.parent)
            if _watch_key not in _watch_latest or _watch_file.name>_watch_latest[_watch_key].name:
                _watch_latest[_watch_key]=_watch_file
        elif (_watch_file.suffix=='.pt' and _watch_file.name.startswith('encoder_')) or _watch_file.suffix in {'.json','.csv','.md','.png','.log','.wandb','.yaml','.txt'} or _watch_file.name.endswith('.csv.gz'):
            _watch_files.append(_watch_file)
    _watch_files.extend(_watch_latest.values())
_watch_inventory=[]
for _watch_file in sorted(_watch_files):
    try:
        _watch_before=_watch_file.stat()
        _watch_hash=None
        if _watch_file.suffix in ('.pt','.wandb') or _watch_file.name.endswith('.csv.gz'):
            with _watch_file.open('rb') as _watch_stream:
                _watch_hash=hashlib.file_digest(_watch_stream,'sha256').hexdigest()
        _watch_after=_watch_file.stat()
        if (_watch_before.st_size,_watch_before.st_mtime_ns)!=(_watch_after.st_size,_watch_after.st_mtime_ns):
            continue
        _watch_inventory.append({'path':str(_watch_file.relative_to(_watch_root)),
            'size':_watch_after.st_size,'mtime_ns':_watch_after.st_mtime_ns,'sha256':_watch_hash})
    except OSError:
        continue
print(MARKER_LITERAL+json.dumps({'status':_watch_state.get('status'),
    'stage':_watch_state.get('stage'),'updated_at':_watch_state.get('updated_at'),
    'current_training':_watch_state.get('current_training'),
    'attempts':_watch_state.get('attempts',[]),'files':_watch_inventory}))
'''.replace("ROOT_LITERAL",repr(remote_output)).replace("STATE_LITERAL",repr(state_file)).replace("MARKER_LITERAL",repr(MARKER))


def parse_probe(stdout):
    # CLI banners and warnings are intentionally ignored and never logged.
    lines=[line[len(MARKER):] for line in stdout.splitlines() if line.startswith(MARKER)]
    if not lines: raise ValueError("Probe did not return its expected marker")
    value=json.loads(lines[-1])
    if not isinstance(value,dict) or not isinstance(value.get("files"),list):
        raise ValueError("Probe schema invalid")
    for record in value["files"]:
        safe_relative(record["path"])
        if not isinstance(record.get("size"),int) or record["size"]<0:
            raise ValueError("Probe size invalid")
    return value


class Watcher:
    def __init__(self,cli,session,remote_output,output,job_start,hours=4,grace_minutes=15,
                 poll_seconds=90,command_timeout=120,state_file="job_state.json"):
        self.cli=str(cli);self.session=session;self.remote_output=remote_output.rstrip("/")
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.artifacts=self.output/"artifacts";self.artifacts.mkdir(exist_ok=True)
        self.poll_seconds=poll_seconds;self.command_timeout=command_timeout
        remote_probe_code(remote_output,state_file)  # validate before monitoring
        self.state_file=state_file
        self.started=parse_start(job_start);self.hard_stop=self.started+hours*3600+grace_minutes*60
        self.grace=grace_minutes*60
        if hours<=0 or grace_minutes<0 or poll_seconds<1: raise ValueError("Invalid time bounds")
        self.status_path=self.output/"watcher_status.json"
        self.index_path=self.output/"download_manifest.json"
        self.index=json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.state={"status":"initializing","pid":os.getpid(),"session":session,
            "job_started_at":job_start,"hard_stop_at":datetime.fromtimestamp(self.hard_stop,timezone.utc).isoformat(),
            "remote_output":self.remote_output,"runtime_stop_confirmed":False,
            "limitation":"Local watcher requires the laptop to remain awake and online; cloud process deadline is separate."}
        self.terminal_seen=None
        self.failures=0

    def event(self,name,**details):
        record={"time":utc_now(),"event":name,**details}
        with (self.output/"watcher_events.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        # Only program-owned safe fields, never raw subprocess output/errors.
        print(json.dumps(record),flush=True)

    def persist(self):
        self.state["updated_at"]=utc_now()
        atomic_json(self.status_path,self.state)
        atomic_json(self.index_path,self.index)

    def command(self,args,input_code=None,timeout=None):
        try:
            result=subprocess.run([self.cli,"--logtostderr",*args],input=input_code,
                text=True,capture_output=True,timeout=timeout or self.command_timeout,check=False)
            return {"ok":result.returncode==0,"code":result.returncode,"stdout":result.stdout}
        except subprocess.TimeoutExpired:
            return {"ok":False,"code":"timeout","stdout":""}
        except OSError:
            return {"ok":False,"code":"local_process_error","stdout":""}

    def probe(self):
        result=self.command(["exec","--session",self.session,"--timeout","45"],
                            input_code=remote_probe_code(self.remote_output,self.state_file))
        if not result["ok"]: return None,result["code"]
        try:return parse_probe(result["stdout"]),None
        except (ValueError,KeyError,TypeError):return None,"invalid_probe"

    def download(self,record,timeout=None):
        rel=safe_relative(record["path"])
        old=self.index.get(str(rel))
        identity={k:record[k] for k in ("size","mtime_ns","sha256")}
        target=self.artifacts/Path(*rel.parts)
        if old==identity and target.is_file() and target.stat().st_size==record["size"]:
            return True
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_suffix(target.suffix+".part")
        result=self.command(["download","--session",self.session,
            self.remote_output+"/"+str(rel),str(part)],timeout=timeout or max(self.command_timeout,240))
        if not result["ok"] or not part.exists():
            part.unlink(missing_ok=True)
            self.event("download_retry_needed",file=str(rel),reason=result["code"])
            return False
        if record["sha256"]:
            with part.open("rb") as f:actual=hashlib.file_digest(f,"sha256").hexdigest()
            if actual!=record["sha256"] or part.stat().st_size!=record["size"]:
                part.unlink(missing_ok=True)
                self.event("download_retry_needed",file=str(rel),reason="checkpoint_integrity_mismatch")
                return False
        elif part.suffixes[-2:-1]==[".json"]:
            try:json.loads(part.read_text())
            except (json.JSONDecodeError,OSError):
                part.unlink(missing_ok=True)
                self.event("download_retry_needed",file=str(rel),reason="incomplete_json")
                return False
        part.replace(target)
        self.index[str(rel)]=identity
        if rel.name.startswith("step_") and rel.suffix==".pt":
            for prior in list(self.index):
                old=PurePosixPath(prior)
                if old.parent==rel.parent and old.name.startswith("step_") and old.suffix==".pt" and old.name<rel.name:
                    (self.artifacts/Path(*old.parts)).unlink(missing_ok=True)
                    del self.index[prior]
                    self.event("retired_older_full_checkpoint",file=prior)
        self.event("downloaded",file=str(rel),bytes=target.stat().st_size)
        return True

    def sync(self,probe,stop_at):
        records=probe["files"]
        # Full resumable checkpoints take precedence if time/network is scarce.
        records=sorted(records,key=lambda r:(0 if PurePosixPath(r["path"]).name.startswith("step_") else
            1 if PurePosixPath(r["path"]).name.startswith("encoder_") else 2,r["path"]))
        complete=True
        for record in records:
            if time.time()>=stop_at:
                complete=False;break
            timeout=max(1,min(max(self.command_timeout,240),stop_at-time.time()))
            complete=self.download(record,timeout=timeout) and complete
        self.persist()
        return complete

    def archive(self):
        destination=self.output/"run_artifacts.tar.gz"
        part=destination.with_suffix(".tar.gz.part")
        # Archive retained artifacts only: all encoder exports, latest full
        # checkpoint per variant, and collected reports. No auth/cache files.
        with tarfile.open(part,"w:gz",compresslevel=1) as archive:
            archive.add(self.artifacts,arcname="artifacts")
            archive.add(self.index_path,arcname="download_manifest.json")
        part.replace(destination)
        self.state["archive"]=str(destination)
        self.event("archived",bytes=destination.stat().st_size)

    def stop(self,reason):
        self.state["stop_reason"]=reason
        self.state["status"]="stopping_runtime"
        self.persist()
        for attempt in range(1,7):
            result=self.command(["stop","--session",self.session])
            if result["ok"] and "Session terminated." in result["stdout"]:
                self.state.update(status="runtime_stopped",runtime_stop_confirmed=True)
                self.event("runtime_stop_confirmed",reason=reason)
                self.persist();return True
            if result["ok"] and "not found" in result["stdout"]:
                self.state.update(status="runtime_release_unverified",runtime_stop_confirmed=False)
                self.event("runtime_release_unverified",reason="session_not_in_cli")
                self.persist();return False
            self.event("runtime_stop_retry",attempt=attempt,reason=result["code"])
            self.persist()
            time.sleep(min(30*attempt,120))
        self.state.update(status="runtime_stop_needs_attention",runtime_stop_confirmed=False)
        self.event("runtime_stop_needs_attention",reason="six_official_cli_attempts_failed")
        self.persist();return False

    def finish(self,reason):
        # All remote downloads have finished before release. Local compression
        # happens after the stop request, so it cannot extend GPU idle time.
        stopped=self.stop(reason)
        self.archive()
        self.persist()
        return 0 if stopped else 2

    def run(self):
        # Prevent two local processes issuing simultaneous downloads/stops.
        with (self.output/"watcher.lock").open("w") as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit("A watcher already owns this output directory")
            (self.output/"watcher.pid").write_text(str(os.getpid())+"\n")
            self.state["status"]="monitoring";self.persist();self.event("watcher_started")
            while True:
                probe,error=self.probe()
                now=time.time()
                if probe is None:
                    self.failures+=1
                    self.state["consecutive_probe_failures"]=self.failures
                    self.event("probe_retry",reason=error,failures=self.failures)
                    self.persist()
                    # A transient probe failure alone never stops the runtime.
                    if now>=self.hard_stop:
                        self.state["artifact_sync_complete"]=False
                        return self.finish("hard_deadline_reached_during_network_failure")
                    time.sleep(min(self.poll_seconds*min(self.failures,3),300,max(1,self.hard_stop-now)))
                    continue
                self.failures=0
                self.state.update(job_status=probe["status"],job_stage=probe.get("stage"),
                    job_updated_at=probe.get("updated_at"),current_training=probe.get("current_training"),
                    consecutive_probe_failures=0)
                if probe["status"] in TERMINAL and self.terminal_seen is None:
                    self.terminal_seen=now
                    self.event("job_terminal",status=probe["status"],stage=probe.get("stage"))
                terminal_stop=min(self.hard_stop,self.terminal_seen+self.grace) if self.terminal_seen else self.hard_stop
                complete=self.sync(probe,terminal_stop)
                self.state["artifact_sync_complete"]=complete
                if (self.terminal_seen is not None and complete) or time.time()>=terminal_stop:
                    reason="job_terminal" if self.terminal_seen is not None else "hard_deadline_reached"
                    return self.finish(reason)
                self.persist()
                time.sleep(min(self.poll_seconds,max(1,terminal_stop-time.time())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli",default=str(DEFAULT_CLI))
    parser.add_argument("--session",default="fjepa-pilot-20260917")
    parser.add_argument("--remote-output",default="/content/fjepa-run")
    parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    parser.add_argument("--job-start",required=True)
    parser.add_argument("--hours",type=float,default=4)
    parser.add_argument("--grace-minutes",type=float,default=15)
    parser.add_argument("--poll-seconds",type=float,default=90)
    parser.add_argument("--command-timeout",type=float,default=120)
    parser.add_argument("--state-file",default="job_state.json")
    args=parser.parse_args()
    return Watcher(**vars(args)).run()


if __name__=="__main__":raise SystemExit(main())
