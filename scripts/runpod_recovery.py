"""Durable local Runpod recovery, independent of the benchmark controller.

Callers must hold ``exclusive_lock`` across journal operations and the entire
lifetime of a run. In particular, persist ``mark_sent`` before issuing POST.
"""
# Local crash-recovery path used by runpod_benchmark.py before provisioning:
# not-sent (no POST) -> may-have-been-sent (search by exact name) -> created
# (Pod ID recorded). The journal tracks deletion confirmations across restarts;
# unresolved records block new paid runs, even if they belong to another account.

from __future__ import annotations

import contextlib
import email.utils
import errno
import fcntl
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class RecoveryError(RuntimeError):
    """The journal or recovery operation cannot be performed safely."""


class LockError(RecoveryError):
    """Another local controller holds the recovery lock."""


class APIError(RecoveryError):
    """An API response could not be used; response bodies are never exposed."""

    def __init__(self, message: str, status: int | None = None):
        """Store a safe error message and the HTTP status, when known."""
        super().__init__(message)
        self.status = status


def _mkdir_private(path: Path) -> None:
    """Create a private state directory without changing existing permissions."""
    # Existing parent directories belong to the operator; never chmod them.
    if not path.exists():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _sync_directory(path: Path) -> None:
    """Sync directory metadata so journal renames survive a crash."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def exclusive_lock(path: str | Path):
    """Acquire a nonblocking, process-wide advisory lock for one local run."""
    path = Path(path)
    _mkdir_private(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockError("another benchmark or recovery holds the lock") from exc
            raise
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class Journal:
    """Atomic, fsynced, owner-only run records. External locking is required."""

    def __init__(self, path: str | Path):
        """Bind the journal to its on-disk path without reading it yet."""
        self.path = Path(path)

    def _load(self) -> list[dict]:
        """Read and validate the journal, treating a missing file as empty."""
        try:
            with self.path.open("r", encoding="utf-8") as source:
                document = json.load(source)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            raise RecoveryError("cannot read recovery journal") from exc
        if not isinstance(document, dict) or document.get("version") != 1:
            raise RecoveryError("invalid or unsupported recovery journal")
        runs = document.get("runs")
        if not isinstance(runs, list):
            raise RecoveryError("invalid recovery journal records")
        ids, names = set(), set()
        for record in runs:
            if not isinstance(record, dict):
                raise RecoveryError("invalid recovery journal record")
            for field in ("run_id", "name", "account"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise RecoveryError("invalid recovery journal identity")
            if record.get("state") not in ("not-sent", "may-have-been-sent", "created"):
                raise RecoveryError("invalid recovery journal state")
            if record["run_id"] in ids or record["name"] in names:
                raise RecoveryError("duplicate recovery journal identity")
            if record.get("pod_id") is not None and not _valid_id(record["pod_id"]):
                raise RecoveryError("invalid journal pod ID")
            for field in ("pending_pod_ids", "deleted_pod_ids"):
                value = record.get(field, [])
                if not isinstance(value, list) or not all(
                    _valid_id(item) for item in value
                ):
                    raise RecoveryError("invalid journal pod ID list")
            if record["state"] == "not-sent" and (
                record.get("pod_id")
                or record.get("pending_pod_ids")
                or record.get("deleted_pod_ids")
            ):
                raise RecoveryError("not-sent run has recorded pod IDs")
            if record["state"] == "created" and not record.get("pod_id"):
                raise RecoveryError("created run lacks a pod ID")
            ids.add(record["run_id"])
            names.add(record["name"])
        return runs

    def _save(self, records: list[dict]) -> None:
        """Replace the journal atomically and sync the file and parent directory."""
        _mkdir_private(self.path.parent)
        fd, temp_name = tempfile.mkstemp(
            prefix=".runpod-journal-", dir=self.path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                fd = -1
                json.dump(
                    {"version": 1, "runs": records}, output, separators=(",", ":")
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_name, self.path)
            _sync_directory(self.path.parent)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _find(records: list[dict], run_id: str) -> dict:
        """Return the record for a run ID or raise if it no longer exists."""
        for record in records:
            if record["run_id"] == run_id:
                return record
        raise RecoveryError("run not found in recovery journal")

    def records(self) -> list[dict]:
        """Return the validated outstanding records from disk."""
        return self._load()

    def begin(self, run_id: str, name: str, expires_at: float, account: str) -> dict:
        """Persist a unique run record in the not-sent state before Pod creation."""
        if not all(
            isinstance(value, str) and value for value in (run_id, name, account)
        ):
            raise RecoveryError("run identity, name, and account must be nonempty")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
            raise RecoveryError("expiration must be an absolute epoch time")
        if not 0 < expires_at < float("inf"):
            raise RecoveryError("expiration must be a finite positive epoch time")
        records = self._load()
        if any(
            record["run_id"] == run_id or record["name"] == name for record in records
        ):
            raise RecoveryError("run ID and pod name must be unique")
        record = {
            "run_id": run_id,
            "name": name,
            "account": account,
            "expires_at": expires_at,
            "started_at": time.time(),
            "state": "not-sent",
            "pod_id": None,
            "last_error": None,
            "pending_pod_ids": [],
            "deleted_pod_ids": [],
        }
        records.append(record)
        self._save(records)
        return record.copy()

    def mark_sent(self, run_id: str) -> None:
        """Mark the create as potentially sent before making the API request."""
        records = self._load()
        record = self._find(records, run_id)
        if record["state"] != "not-sent":
            raise RecoveryError("create was already sent or may have been sent")
        record["state"] = "may-have-been-sent"
        self._save(records)

    def set_pod_id(self, run_id: str, pod_id: str) -> None:
        """Record the created Pod ID as pending deletion."""
        if not _valid_id(pod_id):
            raise RecoveryError("invalid pod ID")
        records = self._load()
        record = self._find(records, run_id)
        if record["state"] == "not-sent":
            raise RecoveryError("cannot persist a pod before marking create sent")
        if record.get("pod_id") not in (None, pod_id):
            raise RecoveryError("pod ID already recorded for this run")
        record["pod_id"] = pod_id
        record["state"] = "created"
        if pod_id not in record.setdefault("pending_pod_ids", []):
            record["pending_pod_ids"].append(pod_id)
        self._save(records)

    def remove(self, run_id: str) -> None:
        """Remove a run record from the journal."""
        records = self._load()
        record = self._find(records, run_id)
        records.remove(record)
        self._save(records)

    def attest(self, run_id: str, name: str, account: str) -> None:
        """Clear only the exact record explicitly identified by the operator."""
        records = self._load()
        record = self._find(records, run_id)
        if record["name"] != name or record["account"] != account:
            raise RecoveryError("attestation must match the exact pod name and account")
        records.remove(record)
        self._save(records)

    def _update(self, run_id: str, **fields) -> None:
        """Persist updated fields for an existing recovery record."""
        records = self._load()
        self._find(records, run_id).update(fields)
        self._save(records)


def _valid_id(value: object) -> bool:
    """Check that an ID is a nonempty string without surrounding whitespace."""
    return isinstance(value, str) and bool(value) and value.strip() == value


def _retry_delay(value: str | None, attempt: int) -> float:
    """Parse Retry-After or fall back to exponential backoff in seconds."""
    if value:
        try:
            seconds = float(value)
            if seconds >= 0 and seconds < float("inf"):
                return seconds
        except ValueError:
            try:
                date = email.utils.parsedate_to_datetime(value)
                return max(0.0, date.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2**attempt)


class RunpodAPI:
    """Minimal REST client with bounded safe retries and no POST retries."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.runpod.io/v2",
        timeout: float = 20,
        max_attempts: int = 4,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
    ):
        """Configure authenticated REST access, timeouts, and retry limits."""
        if not api_key or max_attempts < 1 or timeout <= 0:
            raise ValueError("API key, positive timeout and attempts are required")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._opener = opener
        self._sleep = sleep

    def request(self, method: str, path: str, body: dict | None = None) -> object:
        """Send a REST request and decode its JSON, retrying only safe methods."""
        method = method.upper()
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("API paths must be relative to the configured endpoint")
        data = (
            None if body is None else json.dumps(body, separators=(",", ":")).encode()
        )
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        # Never retry a create: the first POST may have succeeded despite a
        # lost response. Reconcile against the journaled name instead.
        safe_to_retry = method in ("GET", "DELETE")
        for attempt in range(self.max_attempts):
            delay = 2**attempt
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = response.read()
                if not payload:
                    return None
                try:
                    return json.loads(payload)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise APIError("invalid JSON in Runpod API response") from exc
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry = safe_to_retry and (status == 429 or 500 <= status <= 599)
                if not retry or attempt + 1 == self.max_attempts:
                    raise APIError(
                        f"Runpod API returned HTTP {status}", status=status
                    ) from None
                delay = (
                    _retry_delay(exc.headers.get("Retry-After"), attempt)
                    if status == 429
                    else delay
                )
            except (urllib.error.URLError, TimeoutError, OSError):
                if not safe_to_retry or attempt + 1 == self.max_attempts:
                    raise APIError("Runpod API network request failed") from None
            # Do not let an extreme server delay hold a controller indefinitely.
            if delay > 60:
                raise APIError("Runpod retry delay exceeds 60 seconds")
            self._sleep(delay)
        raise APIError("Runpod API attempts exhausted")

    def list_pods(self) -> list[dict]:
        """Return a validated Pod list only after fetching every page."""
        pods = []
        cursor = None
        seen = set()
        while True:
            path = (
                "/pods"
                if cursor is None
                else "/pods?" + urllib.parse.urlencode({"cursor": cursor})
            )
            page = self.request("GET", path)
            if not isinstance(page, dict) or not isinstance(page.get("pods"), list):
                raise APIError("invalid Runpod pod list")
            pagination = page.get("pagination")
            if not isinstance(pagination, dict) or not isinstance(
                pagination.get("hasNextPage"), bool
            ):
                raise APIError("missing Runpod pagination metadata")
            for pod in page["pods"]:
                if (
                    not isinstance(pod, dict)
                    or not _valid_id(pod.get("id"))
                    or not isinstance(pod.get("name"), str)
                ):
                    raise APIError("invalid pod identity in Runpod list")
                pods.append(pod)
            if not pagination["hasNextPage"]:
                return pods
            cursor = pagination.get("nextCursor")
            if not _valid_id(cursor) or cursor in seen:
                raise APIError("invalid or repeated Runpod pagination cursor")
            seen.add(cursor)

    def delete_pod(self, pod_id: str) -> bool:
        """Return True on deletion and False if a 404 leaves it unconfirmed."""
        if not _valid_id(pod_id):
            raise RecoveryError("invalid pod ID")
        try:
            self.request("DELETE", "/pods/" + urllib.parse.quote(pod_id, safe=""))
            return True
        except APIError as exc:
            if exc.status == 404:
                return False
            raise


def reconcile(journal: Journal, api: RunpodAPI, account: str) -> list[dict]:
    """Delete exact-name matches and verified IDs; return all unresolved runs.

    A missing ambiguous create remains unresolved, even after an empty list.
    Failed or 404 deletes remain recorded and block future paid runs. Only a
    confirmed deletion or explicit operator attestation resolves such a record.
    """
    if not isinstance(account, str) or not account:
        raise RecoveryError("authenticated account identity is required")
    records = journal.records()
    for record in records:
        if record["account"] == account and record["state"] == "not-sent":
            journal.remove(record["run_id"])
    records = journal.records()
    eligible = [record for record in records if record["account"] == account]
    if not eligible:
        return records
    # A complete listing discovers exact-name matches, including Pods whose
    # create response was lost. If listing fails, still try known IDs below,
    # but retain the record because there could be undiscovered matches.
    listing_error = None
    try:
        pods = api.list_pods()
    except (APIError, OSError) as exc:
        pods = []
        listing_error = f"pod listing failed: {exc}"
    for record in eligible:
        run_id = record["run_id"]
        pending = set(record.get("pending_pod_ids", []))
        deleted = set(record.get("deleted_pod_ids", []))
        if record.get("pod_id"):
            pending.add(record["pod_id"])
        matches = {pod["id"] for pod in pods if pod["name"] == record["name"]}
        pending.update(matches)
        pending.difference_update(deleted)
        # Record every target before deletion so a crash cannot lose a match.
        fields = {"pending_pod_ids": sorted(pending)}
        if matches and not record.get("pod_id"):
            fields.update(pod_id=min(matches), state="created")
        journal._update(run_id, **fields)
        if not pending and not deleted:
            journal._update(
                run_id,
                last_error=listing_error
                or "create outcome remains ambiguous; no matching pod visible",
            )
            continue
        errors = [listing_error] if listing_error else []
        for pod_id in sorted(pending.copy()):
            try:
                confirmed = api.delete_pod(pod_id)
            except (APIError, OSError) as exc:
                errors.append(f"termination of {pod_id} failed: {exc}")
                continue
            if not confirmed:
                errors.append(
                    f"termination of {pod_id} unconfirmed (404 or inaccessible)"
                )
                continue
            pending.remove(pod_id)
            deleted.add(pod_id)
            # Persist each success so a later recovery doesn't mistake a 404
            # for an unconfirmed deletion of an already-confirmed ID.
            journal._update(
                run_id, pending_pod_ids=sorted(pending), deleted_pod_ids=sorted(deleted)
            )
        if pending or listing_error:
            journal._update(
                run_id,
                last_error="; ".join(errors) or "termination remains unconfirmed",
            )
        else:
            journal.remove(run_id)
    return journal.records()
