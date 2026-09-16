import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from voicehub.hub_transport import _FileLock


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-liveness recovery")
def test_crashed_downloader_lock_is_recovered_without_stale_delay(tmp_path):
    path = tmp_path / "weights.lock"
    code = (
        "from pathlib import Path; import sys,time; from voicehub.hub_transport import _FileLock; "
        "lock=_FileLock(Path(sys.argv[1])); lock.__enter__(); print('locked',flush=True); time.sleep(60)")
    process = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        process.kill()
        process.wait(timeout=5)
        assert path.exists()
        with _FileLock(path, timeout=1, stale_after=86_400):
            assert f":{os.getpid()}:" in path.read_text()
        assert not path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-liveness recovery")
def test_live_downloader_is_not_evicted_when_lock_is_old(tmp_path):
    path = tmp_path / "weights.lock"
    with _FileLock(path):
        old = time.time() - 172_800
        os.utime(path, (old, old))
        with pytest.raises(TimeoutError):
            with _FileLock(path, timeout=0.05, stale_after=1):
                pytest.fail("A live downloader lost its lock")


def test_other_host_pid_is_not_treated_as_a_local_owner(tmp_path):
    path = tmp_path / "weights.lock"
    owner = "another-host:99999999:token"
    path.write_text(owner)
    with pytest.raises(TimeoutError):
        with _FileLock(path, timeout=0.05, stale_after=0):
            pytest.fail("Cannot establish liveness of a remote cache owner")
    assert path.read_text() == owner


def test_legacy_or_incomplete_lock_keeps_age_based_recovery(tmp_path):
    path = tmp_path / "weights.lock"
    path.write_text("")
    old = time.time() - 100
    os.utime(path, (old, old))
    with _FileLock(path, timeout=0.5, stale_after=1):
        assert path.read_text()
