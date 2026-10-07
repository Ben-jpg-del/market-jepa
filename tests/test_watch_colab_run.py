import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("watch_colab",Path(__file__).resolve().parents[1] / "scripts" / "colab" / "watch_colab_run.py")
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


def watcher(tmp_path):
    return watch.Watcher("/unused/colab","fixture","/content/fjepa-run",tmp_path,
        "2099-01-01T00:00:00+00:00",poll_seconds=1)


def test_remote_probe_is_read_only_and_keeps_latest_full_checkpoint(tmp_path):
    run=tmp_path/"remote";variant=run/"variant_C_seed42";variant.mkdir(parents=True)
    for name in ("step_000100.pt","step_000500.pt","encoder_step_000100.pt","encoder_step_000500.pt"):
        (variant/name).write_bytes(name.encode())
    (run/"job_state.json").write_text(json.dumps({"status":"running","stage":"screen_C"}))
    (run/"secret.env").write_text("DO_NOT_READ")
    before={str(p):p.stat().st_mtime_ns for p in run.rglob("*")}
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(watch.remote_probe_code(str(run)),{})
    probe=watch.parse_probe(stream.getvalue())
    paths=[r["path"] for r in probe["files"]]
    assert "variant_C_seed42/step_000500.pt" in paths
    assert "variant_C_seed42/step_000100.pt" not in paths
    assert len([p for p in paths if "encoder_step" in p])==2
    assert "secret.env" not in paths
    assert before=={str(p):p.stat().st_mtime_ns for p in run.rglob("*")}


def test_probe_rejects_path_traversal():
    with pytest.raises(ValueError,match="Unsafe"):
        watch.parse_probe(watch.MARKER+json.dumps({"files":[{"path":"../secret","size":3}]}))


def test_expanded_job_terminal_state_is_detected(tmp_path):
    (tmp_path/"expanded_job_state.json").write_text(json.dumps({"status":"completed","stage":"done"}))
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(watch.remote_probe_code(str(tmp_path),"expanded_job_state.json"),{})
    assert watch.parse_probe(stream.getvalue())["status"]=="completed"
    with pytest.raises(ValueError):
        watch.remote_probe_code(str(tmp_path),"../state.json")


def test_compressed_predictions_are_hashed_and_scratch_cache_is_excluded(tmp_path):
    (tmp_path/'predictions.csv.gz').write_bytes(b'compressed-test-content')
    scratch=tmp_path/'expanded_evaluation_cache';scratch.mkdir()
    (scratch/'manifest.json').write_text('{}')
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(watch.remote_probe_code(str(tmp_path)),{})
    files=watch.parse_probe(stream.getvalue())['files']
    assert [r['path'] for r in files]==['predictions.csv.gz']
    assert files[0]['sha256']==hashlib.sha256(b'compressed-test-content').hexdigest()


def test_checkpoint_download_hash_and_latest_retention(tmp_path):
    w=watcher(tmp_path)
    payload=b"valid checkpoint"
    digest=hashlib.sha256(payload).hexdigest()
    def command(args,**kwargs):
        Path(args[-1]).write_bytes(payload)
        return {"ok":True,"code":0,"stdout":"unlogged_cli_secret"}
    w.command=command
    old={"path":"variant_C_seed42/step_000100.pt","size":len(payload),"mtime_ns":1,"sha256":digest}
    new={**old,"path":"variant_C_seed42/step_000500.pt","mtime_ns":2}
    assert w.download(old) and w.download(new)
    assert not (w.artifacts/old["path"]).exists()
    assert (w.artifacts/new["path"]).exists()
    assert old["path"] not in w.index
    bad={**new,"path":"variant_C_seed42/step_001000.pt","sha256":"0"*64}
    assert not w.download(bad)
    assert not (w.artifacts/bad["path"]).exists()
    assert "unlogged_cli_secret" not in (tmp_path/"watcher_events.jsonl").read_text()


def test_stop_requires_positive_confirmation_not_missing_session(tmp_path):
    w=watcher(tmp_path)
    w.command=lambda *a,**kw:{"ok":True,"code":0,"stdout":"[colab] Session 'fixture' not found."}
    assert not w.stop("test")
    assert not w.state["runtime_stop_confirmed"]
    assert w.state["status"]=="runtime_release_unverified"


def test_transient_probe_failure_does_not_trigger_stop(monkeypatch,tmp_path):
    w=watcher(tmp_path)
    probes=iter([(None,"timeout"),({"status":"completed","stage":"done","files":[]},None)])
    w.probe=lambda:next(probes)
    monkeypatch.setattr(watch.time,"sleep",lambda seconds:None)
    reasons=[]
    w.finish=lambda reason:reasons.append(reason) or 0
    assert w.run()==0
    assert reasons==["job_terminal"]
    events=(tmp_path/"watcher_events.jsonl").read_text()
    assert "probe_retry" in events and "job_terminal" in events


def test_finish_stops_before_local_archiving(tmp_path):
    w=watcher(tmp_path);order=[]
    w.stop=lambda reason:order.append("stop") or True
    w.archive=lambda:order.append("archive")
    assert w.finish("test")==0
    assert order==["stop","archive"]
