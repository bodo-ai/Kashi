import io
import json
import os
import stat
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.runpod_recovery import (
    APIError,
    Journal,
    LockError,
    RecoveryError,
    RunpodAPI,
    exclusive_lock,
    reconcile,
)


class Response:
    def __init__(self, value=None, status=200):
        self.value = b"" if value is None else json.dumps(value).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.value


class FakeAPI:
    def __init__(self, pods=(), deletes=None, error=None):
        self.pods = list(pods)
        self.deletes = dict(deletes or {})
        self.error = error
        self.deleted = []

    def list_pods(self):
        if self.error:
            raise APIError(self.error)
        return list(self.pods)

    def delete_pod(self, pod_id):
        self.deleted.append(pod_id)
        result = self.deletes.get(pod_id, True)
        if isinstance(result, Exception):
            raise result
        return result


def create_record(journal, run_id="run-1", name="bench-exact", account="account-1"):
    return journal.begin(run_id, name, 1700000000, account)


def test_journal_persists_identity_and_each_transition_privately(tmp_path):
    path = tmp_path / "private" / "journal.json"
    journal = Journal(path)
    record = create_record(journal)
    assert record["state"] == "not-sent"
    assert record["expires_at"] == 1700000000
    assert record["account"] == "account-1"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    journal.mark_sent("run-1")
    assert Journal(path).records()[0]["state"] == "may-have-been-sent"
    journal.set_pod_id("run-1", "pod-1")
    restored = Journal(path).records()[0]
    assert restored["pod_id"] == "pod-1"
    assert restored["name"] == "bench-exact"
    assert restored["state"] == "created"


def test_journal_syncs_file_before_replace_and_directory_after(tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    create_record(Journal(tmp_path / "journal.json"))
    assert events == ["file", "replace", "directory"]


def test_invalid_journal_state_cannot_discard_a_record(tmp_path):
    path = tmp_path / "journal.json"
    journal = Journal(path)
    create_record(journal)
    content = json.loads(path.read_text())
    content["runs"][0]["pod_id"] = "pod-1"
    path.write_text(json.dumps(content))
    api = FakeAPI()
    with pytest.raises(RecoveryError):
        reconcile(journal, api, "account-1")
    assert not api.deleted


def test_duplicate_run_identity_and_invalid_transitions_are_rejected(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    with pytest.raises(RecoveryError):
        create_record(journal)
    with pytest.raises(RecoveryError):
        create_record(journal, "another", "bench-exact")
    with pytest.raises(RecoveryError):
        journal.set_pod_id("run-1", "pod-1")
    with pytest.raises(RecoveryError):
        journal.mark_sent("missing")


def test_lock_is_exclusive_and_released_after_exit(tmp_path):
    path = tmp_path / "lock"
    with exclusive_lock(path):
        with pytest.raises(LockError), exclusive_lock(path):
            pass
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with exclusive_lock(path):
        pass


def test_lock_prevents_simultaneous_process_and_crash_releases_it(tmp_path):
    path = tmp_path / "lock"
    script = """from scripts.runpod_recovery import exclusive_lock
import os, sys
with exclusive_lock(sys.argv[1]):
 print('locked', flush=True)
 sys.stdin.read(1)
 os._exit(9)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(LockError), exclusive_lock(path):
            pass
        proc.stdin.write("x")
        proc.stdin.flush()
        assert proc.wait(timeout=5) == 9
        with exclusive_lock(path):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_list_pods_walks_every_cursor_and_encodes_it():
    requests = []
    pages = [
        {
            "pods": [{"id": "one", "name": "first"}],
            "pagination": {"nextCursor": "a+/=", "hasNextPage": True},
        },
        {
            "pods": [{"id": "two", "name": "second"}],
            "pagination": {"nextCursor": None, "hasNextPage": False},
        },
    ]

    def open_request(request, timeout):
        requests.append((request.full_url, request.get_header("Authorization")))
        return Response(pages.pop(0))

    api = RunpodAPI("key-123", opener=open_request)
    assert [pod["id"] for pod in api.list_pods()] == ["one", "two"]
    assert requests == [
        ("https://api.runpod.io/v2/pods", "Bearer key-123"),
        ("https://api.runpod.io/v2/pods?cursor=a%2B%2F%3D", "Bearer key-123"),
    ]


@pytest.mark.parametrize(
    "pagination",
    [
        None,
        {},
        {"hasNextPage": True, "nextCursor": None},
        {"hasNextPage": "false", "nextCursor": None},
    ],
)
def test_list_pods_rejects_incomplete_pagination(pagination):
    api = RunpodAPI(
        "key",
        opener=lambda *_args, **_kwargs: Response(
            {"pods": [], "pagination": pagination}
        ),
    )
    with pytest.raises(APIError):
        api.list_pods()


def test_list_pods_rejects_repeated_cursor():
    api = RunpodAPI(
        "key",
        opener=lambda *_args, **_kwargs: Response(
            {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "same"}}
        ),
    )
    with pytest.raises(APIError):
        api.list_pods()


def http_error(request, status, headers=None):
    return urllib.error.HTTPError(
        request.full_url,
        status,
        "failed",
        headers or {},
        io.BytesIO(b"sensitive error body"),
    )


def test_rate_limit_retries_honor_retry_after_for_get_and_delete():
    calls = []
    sleeps = []

    def opener(request, timeout):
        calls.append(request.method)
        if len(calls) in (1, 3):
            raise http_error(request, 429, {"Retry-After": "3"})
        return Response({"ok": True} if request.method == "GET" else None, status=204)

    api = RunpodAPI("key", opener=opener, sleep=sleeps.append)
    assert api.request("GET", "/account") == {"ok": True}
    assert api.delete_pod("pod-1")
    assert calls == ["GET", "GET", "DELETE", "DELETE"]
    assert sleeps == [3, 3]


def test_repeated_api_failure_has_bounded_retries_and_sanitized_error():
    calls = []
    sleeps = []

    def opener(request, timeout):
        calls.append(request.method)
        raise http_error(request, 503)

    api = RunpodAPI("secret-key", max_attempts=3, opener=opener, sleep=sleeps.append)
    with pytest.raises(APIError) as error:
        api.request("GET", "/account")
    assert calls == ["GET", "GET", "GET"]
    assert sleeps == [1, 2]
    assert "secret-key" not in str(error.value)
    assert "sensitive error body" not in str(error.value)


def test_long_retry_after_fails_without_retrying_too_early():
    calls = []

    def opener(request, timeout):
        calls.append(request.method)
        raise http_error(request, 429, {"Retry-After": "3600"})

    api = RunpodAPI(
        "key", opener=opener, sleep=lambda _: pytest.fail("must not sleep an hour")
    )
    with pytest.raises(APIError):
        api.list_pods()
    assert calls == ["GET"]


def test_create_is_never_retried_after_ambiguous_failure():
    calls = []

    def opener(request, timeout):
        calls.append(request.method)
        raise urllib.error.URLError("network lost")

    api = RunpodAPI("secret-key", opener=opener, sleep=lambda _: None)
    with pytest.raises(APIError) as error:
        api.request("POST", "/pods", {"name": "bench"})
    assert calls == ["POST"]
    assert "secret-key" not in str(error.value)


def test_delete_404_is_unconfirmed_and_does_not_remove_journal(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    journal.set_pod_id("run-1", "pod-1")
    api = FakeAPI(deletes={"pod-1": False})
    remaining = reconcile(journal, api, "account-1")
    assert remaining[0]["pod_id"] == "pod-1"
    assert remaining[0]["last_error"]
    assert Journal(tmp_path / "journal.json").records()


def test_reconcile_removes_not_sent_without_deleting_anything(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    api = FakeAPI([{"name": "bench-exact", "id": "unrelated"}])
    assert reconcile(journal, api, "account-1") == []
    assert api.deleted == []


def test_ambiguous_create_remains_unresolved_until_delayed_pod_is_found(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    api = FakeAPI()
    assert reconcile(journal, api, "account-1")[0]["name"] == "bench-exact"
    assert not api.deleted
    api.pods = [
        {"name": "bench-exact-extra", "id": "other"},
        {"name": "bench-exact", "id": "late"},
    ]
    assert reconcile(journal, api, "account-1") == []
    assert api.deleted == ["late"]


def test_reconcile_deletes_all_exact_matches_and_verified_id_only(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    journal.set_pod_id("run-1", "verified")
    api = FakeAPI(
        [
            {"name": "bench-exact", "id": "first"},
            {"name": "bench-exact", "id": "second"},
            {"name": "bench-exact-other", "id": "unrelated"},
        ]
    )
    assert reconcile(journal, api, "account-1") == []
    assert set(api.deleted) == {"verified", "first", "second"}


def test_account_change_is_safe_and_listing_failure_still_deletes_verified_id(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    journal.set_pod_id("run-1", "pod-1")
    api = FakeAPI(error="list unavailable")
    remaining = reconcile(journal, api, "another-account")
    assert remaining[0]["account"] == "account-1"
    assert api.deleted == []
    remaining = reconcile(journal, api, "account-1")
    assert remaining[0]["last_error"]
    assert api.deleted == ["pod-1"]
    api.error = None
    assert reconcile(journal, api, "account-1") == []
    assert api.deleted == ["pod-1"]


def test_failed_deletion_does_not_forget_previously_confirmed_matches(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    api = FakeAPI(
        [{"name": "bench-exact", "id": "one"}, {"name": "bench-exact", "id": "two"}],
        deletes={"two": False},
    )
    assert reconcile(journal, api, "account-1")
    assert set(api.deleted) == {"one", "two"}
    api.deleted.clear()
    api.pods = []
    api.deletes = {"two": True}
    assert reconcile(Journal(tmp_path / "journal.json"), api, "account-1") == []
    assert api.deleted == ["two"]


def test_attestation_requires_matching_exact_name_and_account(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    create_record(journal)
    journal.mark_sent("run-1")
    with pytest.raises(RecoveryError):
        journal.attest("run-1", "bench", "account-1")
    with pytest.raises(RecoveryError):
        journal.attest("run-1", "bench-exact", "other-account")
    assert len(journal.records()) == 1
    journal.attest("run-1", "bench-exact", "account-1")
    assert journal.records() == []


def test_http_delete_404_is_not_deletion_confirmation():
    def opener(request, timeout):
        raise http_error(request, 404)

    assert not RunpodAPI("key", opener=opener).delete_pod("pod")


def test_create_sent_state_survives_abrupt_controller_exit(tmp_path):
    path = tmp_path / "journal.json"
    script = """from scripts.runpod_recovery import Journal
import os,sys
j=Journal(sys.argv[1])
j.begin('run', 'bench-abrupt', 1700000000, 'account')
j.mark_sent('run')
os._exit(13)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert result.returncode == 13
    journal = Journal(path)
    assert reconcile(journal, FakeAPI(), "account")[0]["state"] == "may-have-been-sent"
    api = FakeAPI([{"name": "bench-abrupt", "id": "lost"}])
    assert reconcile(journal, api, "account") == []
    assert api.deleted == ["lost"]
