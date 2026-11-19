import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "runpod_benchmark", Path(__file__).parents[2] / "scripts" / "runpod_benchmark.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_download_checks_checksum_and_saves_exact_bytes(tmp_path):
    data = b"benchmark artifact"

    def get(offset, size):
        return data[offset : offset + min(size, 4)]

    result = controller.download_artifact(
        get, len(data), hashlib.sha256(data).hexdigest(), tmp_path / "result.tar.gz"
    )
    assert result.read_bytes() == data


def test_download_rejects_bad_checksum_without_publishing(tmp_path):
    target = tmp_path / "result.tar.gz"
    with pytest.raises(ValueError, match="checksum"):
        controller.download_artifact(
            lambda offset, size: b"bad"[offset : offset + size], 3, "0" * 64, target
        )
    assert not target.exists()


def test_positive_workload_validation():
    with pytest.raises(SystemExit):
        controller.parse_args(["run", "--prompts", "0"])
    with pytest.raises(SystemExit):
        controller.parse_args(["run", "--seed", "-1"])


def test_failed_worker_preserves_artifact_and_terminates(tmp_path, monkeypatch):
    import json

    data = b"partial diagnostics"
    args = controller.parse_args(["run", "--output-dir", str(tmp_path)])

    class Journal:
        def begin(self, *args):
            pass

        def mark_sent(self, *args):
            pass

        def set_pod_id(self, *args):
            pass

        def remove(self, *args):
            pass

    class API:
        deleted = False

        def request(self, method, path, body=None):
            if method == "POST":
                return {"id": "pod1"}
            return {"status": "RUNNING"}

        def delete_pod(self, pod_id):
            self.deleted = pod_id == "pod1"
            return self.deleted

    api = API()

    def worker(pod, secret, path, deadline):
        if path == "/status":
            return json.dumps(
                {
                    "state": "failed",
                    "error": "model failure",
                    "artifact_size": len(data),
                    "artifact_sha256": hashlib.sha256(data).hexdigest(),
                }
            ).encode()
        return data

    monkeypatch.setattr(controller, "worker_request", worker)
    with pytest.raises(RuntimeError, match="model failure"):
        controller.run_benchmark(
            args, api, Journal(), "account", "AMD Instinct MI300X OAM"
        )
    assert api.deleted
    assert next(tmp_path.glob("*.partial.tar.gz")).read_bytes() == data
    sidecar = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert sidecar["cleanup_confirmed"] is True
    assert sidecar["benchmark_success"] is False


def test_unconfirmed_cleanup_is_not_success(tmp_path, monkeypatch):
    import json

    data = b"results"
    args = controller.parse_args(["run", "--output-dir", str(tmp_path)])

    class Journal:
        def begin(self, *args):
            pass

        def mark_sent(self, *args):
            pass

        def set_pod_id(self, *args):
            pass

        def records(self):
            return []

    class API:
        def request(self, method, path, body=None):
            return {"id": "pod1"} if method == "POST" else {"status": "RUNNING"}

        def delete_pod(self, pod_id):
            return False

    def worker(pod, secret, path, deadline):
        if path == "/status":
            return json.dumps(
                {
                    "state": "done",
                    "artifact_size": len(data),
                    "artifact_sha256": hashlib.sha256(data).hexdigest(),
                }
            ).encode()
        return data

    monkeypatch.setattr(controller, "worker_request", worker)
    assert (
        controller.run_benchmark(
            args, API(), Journal(), "account", "AMD Instinct MI300X OAM"
        )
        == 1
    )
    sidecar = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert sidecar["cleanup_confirmed"] is False


@pytest.mark.parametrize("availability", ["NONE", "HIGH"])
def test_rejected_create_reports_current_capacity_and_clears_journal(
    tmp_path, capsys, availability
):
    api_error = controller.recovery.APIError("Runpod API returned HTTP 400", status=400)

    class API:
        def request(self, method, path, body=None):
            if method == "POST":
                raise api_error
            assert path == "/catalog/gpus?include=AVAILABILITY&product=POD&count=1"
            return {
                "gpus": [
                    {"id": "AMD Instinct MI300X OAM", "availability": availability}
                ]
            }

    journal = controller.recovery.Journal(tmp_path / "journal.json")
    with pytest.raises(controller.recovery.APIError) as error:
        controller.create(
            API(),
            journal,
            "account",
            "run-name",
            1234,
            {"gpu": {"id": "AMD Instinct MI300X OAM"}},
            "run-id",
        )
    assert error.value is api_error
    output = capsys.readouterr().err
    assert availability in output
    assert "does not establish the cause" in output
    assert journal.records() == []


def test_capacity_lookup_failure_preserves_create_error_and_clears_journal(
    tmp_path, capsys
):
    original = controller.recovery.APIError("Runpod API returned HTTP 400", status=400)

    class API:
        def request(self, method, path, body=None):
            if method == "POST":
                raise original
            raise controller.recovery.APIError("catalog inaccessible", status=403)

    journal = controller.recovery.Journal(tmp_path / "journal.json")
    with pytest.raises(controller.recovery.APIError) as error:
        controller.create(API(), journal, "account", "run-name", 1234, {}, "run-id")
    assert error.value is original
    assert "unavailable" in capsys.readouterr().err.lower()
    assert journal.records() == []


def test_ambiguous_create_error_keeps_journal_for_recovery(tmp_path):
    class API:
        def request(self, method, path, body=None):
            raise controller.recovery.APIError("network failure")

    journal = controller.recovery.Journal(tmp_path / "journal.json")
    with pytest.raises(controller.recovery.APIError):
        controller.create(API(), journal, "account", "run-name", 1234, {}, "run-id")
    assert journal.records()[0]["state"] == "may-have-been-sent"


def test_wait_capacity_uses_requested_data_center(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        controller.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )

    class API:
        calls = 0

        def request(self, method, path):
            self.calls += 1
            availability = "NONE" if self.calls == 1 else "LOW"
            return {
                "gpus": [
                    {
                        "id": "AMD Instinct MI300X OAM",
                        "availability": "HIGH",
                        "dataCenters": [
                            {"id": "US-KS-2", "availability": availability}
                        ],
                    }
                ]
            }

    api = API()
    assert (
        controller.wait_for_capacity(api, 120, "US-KS-2", ["AMD Instinct MI300X OAM"])
        == "AMD Instinct MI300X OAM"
    )
    assert api.calls == 2
    assert now[0] > 0


def test_wait_capacity_times_out_without_deployment(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        controller.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )

    class API:
        def request(self, method, path):
            assert method == "GET"
            return {"gpus": [{"id": "AMD Instinct MI300X OAM", "availability": "NONE"}]}

    with pytest.raises(TimeoutError):
        controller.wait_for_capacity(API(), 40, None, ["AMD Instinct MI300X OAM"])
    assert now[0] == 40


def test_selected_gpu_is_sent_to_pod_and_worker(tmp_path, monkeypatch):
    import json

    args = controller.parse_args(
        ["run", "--gpu", "AMD Instinct MI350 OAM", "--output-dir", str(tmp_path)]
    )

    class Journal:
        def begin(self, *args):
            pass

        def mark_sent(self, *args):
            pass

        def set_pod_id(self, *args):
            pass

        def remove(self, *args):
            pass

    class API:
        def request(self, method, path, body=None):
            if method == "POST":
                assert body["gpu"]["id"] == "AMD Instinct MI350 OAM"
                assert (
                    json.loads(body["env"]["BENCH_CONFIG"])["gpu"]
                    == "AMD Instinct MI350 OAM"
                )
                return {"id": "pod1"}
            return {"status": "ERROR"}

        def delete_pod(self, pod_id):
            return True

    with pytest.raises(RuntimeError, match="Pod status"):
        controller.run_benchmark(
            args, API(), Journal(), "account", "AMD Instinct MI350 OAM"
        )


def test_proxy_request_succeeds_when_default_python_agent_is_blocked(monkeypatch):
    import io
    import time
    import urllib.error

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"state":"done"}'

    def proxy(request, timeout):
        if not request.get_header("User-agent"):
            raise urllib.error.HTTPError(
                request.full_url, 403, "blocked", {}, io.BytesIO()
            )
        assert request.get_header("Authorization") == "Bearer secret"
        return Response()

    monkeypatch.setattr(controller.urllib.request, "urlopen", proxy)
    assert (
        controller.worker_request("pod", "secret", "/status", time.time() + 30)
        == b'{"state":"done"}'
    )


def test_gpu_preference_defaults_and_override():
    assert controller.parse_args(["run"]).gpu == [
        "AMD Instinct MI300X OAM",
        "AMD Instinct MI350 OAM",
    ]
    assert controller.parse_args(
        ["run", "--gpu", "AMD Instinct MI350 OAM", "AMD Instinct MI300X OAM"]
    ).gpu == ["AMD Instinct MI350 OAM", "AMD Instinct MI300X OAM"]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("HIGH", "LOW", "AMD Instinct MI300X OAM"),
        ("NONE", "LOW", "AMD Instinct MI350 OAM"),
    ],
)
def test_gpu_selection_follows_preference(first, second, expected):
    class API:
        def request(self, method, path):
            return {
                "gpus": [
                    {"id": "AMD Instinct MI350 OAM", "availability": second},
                    {"id": "AMD Instinct MI300X OAM", "availability": first},
                ]
            }

    assert (
        controller.wait_for_capacity(
            API(), 0, None, ["AMD Instinct MI300X OAM", "AMD Instinct MI350 OAM"]
        )
        == expected
    )


def test_gpu_selection_respects_custom_order_and_data_center():
    class API:
        def request(self, method, path):
            return {
                "gpus": [
                    {
                        "id": "AMD Instinct MI300X OAM",
                        "availability": "HIGH",
                        "dataCenters": [{"id": "DC1", "availability": "NONE"}],
                    },
                    {
                        "id": "AMD Instinct MI350 OAM",
                        "availability": "LOW",
                        "dataCenters": [{"id": "DC1", "availability": "LOW"}],
                    },
                ]
            }

    assert (
        controller.wait_for_capacity(
            API(), 0, "DC1", ["AMD Instinct MI300X OAM", "AMD Instinct MI350 OAM"]
        )
        == "AMD Instinct MI350 OAM"
    )
    assert (
        controller.wait_for_capacity(
            API(), 0, None, ["AMD Instinct MI350 OAM", "AMD Instinct MI300X OAM"]
        )
        == "AMD Instinct MI350 OAM"
    )


def test_unknown_preferred_gpu_capacity_does_not_silently_select_other():
    class API:
        def request(self, method, path):
            return {"gpus": [{"id": "AMD Instinct MI350 OAM", "availability": "HIGH"}]}

    with pytest.raises(RuntimeError, match="capacity information is unavailable"):
        controller.wait_for_capacity(
            API(), 0, None, ["AMD Instinct MI300X OAM", "AMD Instinct MI350 OAM"]
        )


def test_run_does_not_require_watchdog_confirmation_marker(tmp_path, monkeypatch):
    class API:
        def __init__(self, key):
            pass

        def request(self, method, path, body=None):
            if path == "/account":
                return {"id": "account"}
            return {"gpus": [{"id": "AMD Instinct MI300X OAM", "availability": "HIGH"}]}

    monkeypatch.setattr(controller.recovery, "RunpodAPI", API)
    monkeypatch.setattr(
        controller, "run_benchmark", lambda args, api, journal, account, gpu: 0
    )
    assert (
        controller.main(["--api-key", "key", "--state-dir", str(tmp_path), "run"]) == 0
    )
