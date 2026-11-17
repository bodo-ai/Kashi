# Runpod MI300X vLLM Benchmark

**Status:** Proposed

## Intent

Provide a repeatable, end-to-end command that rents one AMD Instinct MI300X OAM Pod on Runpod, serves the selected Qwen3.8-27B FP8 recipe, benchmarks it, optionally captures a GPU profile, downloads a local artifact, and terminates the Pod. The results are for comparing serving performance, not evaluating model quality. Running the command incurs Runpod compute, storage, and network charges.

## Approaches

- **Selected: local Python controller and temporary Pod.** Run the benchmark inside the recipe's ROCm image and retrieve results through an authenticated HTTPS-proxied service. An in-Pod watchdog and local cleanup protect against abandoned runs without requiring a GitHub integration.
- **SSH controller.** Execute and transfer files through SSH. This is straightforward when configured, but requires additional keys and a working SSH daemon; direct SCP also requires a public IP.
- **Runpod Serverless endpoint.** Useful for production serving, but cold starts, scaling, and request routing complicate a controlled single-GPU benchmark and its profiling artifacts.

## Recipe and compatibility

Use the [selected recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json?hardware=mi300x&variant=fp8&features=tool_calling%2Creasoning%2Cspec_decoding%2Ctext_only) as retrieved on 2026-11-16:

```sh
export VLLM_USE_RUST_FRONTEND=1
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --language-model-only
```

The default image is `vllm/vllm-openai-rocm:latest`, the recipe's image. The recipe's hardware label says 8× MI300X, but its rendered serving command specifies tensor parallelism of 1; this benchmark explicitly rents **one** MI300X and follows that command. The recipe requires vLLM >= 0.17.0 and `transformers>=5.8.0`. Check these requirements in the Pod and fail with a diagnostic if unmet; do not silently replace the image or install unpinned packages. Allow an image reference override for a compatible pinned image, and record the actual versions and image reference. Resolve and record an image digest and model revision when available; mark either as unavailable rather than inventing one. `latest` and an unpinned model revision are not reproducible across runs.

The recipe's Docker example requests ROCm devices, privileged execution, `seccomp=unconfined`, and host IPC. Runpod controls container isolation; do not assume every Docker flag can be requested through its Pod API. Check GPU visibility and fail explicitly if the selected Runpod environment cannot run the recipe. A successful MI300X deployment and ROCm profile cannot be inferred from the catalog or offline tests; report them as unverified until observed in an actual benchmark run.

## Interface and lifecycle

- Provide a standalone Python command under `scripts/`, compatible with the repository's supported Python versions. It requires a Runpod API key via `RUNPOD_API_KEY`, with an optional command-line key override. An optional `HF_TOKEN` supports authenticated model downloads. Avoid requiring local GPU, Docker, SSH keys, or a separately deployed service.
- Defaults: one `AMD Instinct MI300X OAM`, one Pod, 100 GB ephemeral container disk, Runpod's default cloud and placement, and a 90-minute overall deadline. Permit overriding the image, disk size, deadline, and data center. Do not select a different GPU or queue indefinitely if capacity is unavailable.
- Use the [Runpod REST API v2](https://docs.runpod.io/api-reference-v2/pods/create-a-pod), with a unique run name, explicit image, GPU and count. Override the container entrypoint to bootstrap the remote worker; use the recipe command inside that container. Poll both Pod state and worker readiness because Pod `RUNNING` alone does not mean vLLM is ready. Report provisioning, model loading, warmup, benchmark, optional profiling, download, and cleanup progress.
- Expose only the worker's control port through Runpod's HTTPS proxy. Generate a fresh high-entropy per-run bearer secret; require it for every control and artifact request, compare it safely, and bind vLLM itself to loopback. The worker must not expose arbitrary shell execution or filesystem access. Long operations run in the background with short status requests; download artifacts in bounded chunks so proxy connection timeouts do not truncate large traces. Verify size and SHA-256 before accepting the downloaded artifact.
- Persist the run identity before creating the Pod and its ID immediately after receiving it. Terminate on every ordinary exit, and reconcile the journal on startup for crashes, ambiguous creates, and inaccessible Pods. A missing or unconfirmed cleanup is an error, never a successful run.
- Exit zero only when the benchmark, requested profile, artifact verification, and Pod termination all succeed. Preserve available partial diagnostics locally on failure; never report missing metrics or traces as a successful run. API keys and bearer secrets must not appear in command output, artifact contents, or diagnostic logs. Treat Runpod Pod environment values as visible to account/API users and document that exposure for an optional HF token.

## Disaster recovery and budget protection

Runpod's documented Pod API has no server-enforced maximum lifetime. The in-Pod watchdog cannot guarantee cleanup if the container never starts or Runpod's API is unreachable; a stray Pod may continue billing until the script is run again or the operator intervenes. The deadline and estimated cost are not hard billing caps.

- Before creating a Pod, write a unique random run ID, exact Pod name, absolute expiration, intended account, and `not-sent` state to an atomic durable local journal with owner-only permissions; synchronize the file and containing directory. Transition durably to `may-have-been-sent` before the create request, then persist the Pod ID when known. The default 90-minute expiration starts before create and is never extended. Do not issue queued deployments or retry an ambiguous create.
- Start an in-Pod watchdog before loading the model. At the absolute deadline it attempts to terminate its own `RUNPOD_POD_ID` using the Pod-scoped `RUNPOD_API_KEY`, retrying transient failures. Do not inject the controller's account key into the Pod. If the Pod-scoped key or watchdog is unavailable, fail the run and have the local controller terminate the Pod. A stopped container or Pod is not confirmed termination.
- On ordinary completion, error, timeout, or interrupt, the controller terminates the Pod and checks the result; termination takes priority over finishing artifact retrieval or profiling after the deadline. There is no detached local watchdog.
- At startup, before a new run, reconcile unresolved journal entries against all pages of the authenticated Runpod Pod list by exact unique name and terminate matching stray Pods, regardless of their deadline. Hold an exclusive local lock while a run is active so a second invocation cannot mistake its Pod for a stray. For ambiguous creates, retain an unresolved record until the outcome is established; an empty list once does not prove a delayed create cannot appear. Delete only the recorded run's exact matches or verified Pod ID, with bounded retries and `Retry-After` handling. A successful DELETE confirms deletion; a 404 alone does not when access may have changed. Block new paid runs while records remain unresolved.
- Provide a recovery command and manual console instructions. On unconfirmed cleanup, report the exact Pod name and ID if known, account, elapsed time, last error, and termination instructions. If API evidence cannot establish the outcome, require explicit operator attestation before clearing the record. Preserve retrieved partial artifacts and never report the run as successful.
- Report the Pod's hourly rate and estimated cost when available, clearly excluding unknown charges and overrun. Recommend a dedicated prepaid account with auto-pay disabled and limited credits as an additional budget backstop. Create no persistent or network volumes.

## Benchmark

Run `vllm bench serve` inside the Pod against `127.0.0.1`, using the OpenAI chat endpoint, the recipe model, streaming, and a seeded synthetic random text dataset. Defaults: 128 measured prompts, 5 warmup requests, 512 input tokens, 256 requested output tokens, maximum concurrency 8, and seed 0. Expose prompt count, warmups, input/output lengths, concurrency, and seed as command-line options; validate positive counts and lengths and a nonnegative seed, and record them. Use a fixed output-length setting where supported and report actual token counts, since model behavior and early stopping can change generated lengths.

Save the native benchmark JSON and report request and output-token throughput, completed and failed requests, and mean plus p50/p95/p99 time-to-first-token, time per output token, inter-token latency, and end-to-end latency when supported. Record measurement units and benchmark duration. Request counts and metrics must be taken from actual measurements; a nonzero request failure count makes the run fail while preserving results. Synthetic random text is a throughput workload, not a representative tool-calling or reasoning-quality evaluation; enabling the recipe's parsers does not demonstrate tool-call correctness.

Collect vLLM Prometheus metrics immediately before and after the measured phase. Include the deltas of `vllm:spec_decode_num_accepted_tokens_total` and `vllm:spec_decode_num_draft_tokens_total` and their ratio when the draft delta is positive. If the metrics are absent or the denominator is zero, record the acceptance rate as unavailable with the reason, not zero or a fabricated success.

## Optional profiling

`--profile` adds a **separate** short profiling pass after the unprofiled benchmark; baseline measurements must not be collected while profiling is active. Configure vLLM's PyTorch profiler at server startup and bracket a small, bounded request workload using the documented profile controls. Always stop and flush the profiler before retrieval, including on failure. Save the raw compressed trace and available profiler summary alongside a label identifying the profiled workload. The profiling pass may be slower and its results must not be presented as baseline performance.

Use a ROCm-capable PyTorch profiler configuration; verify that the trace contains GPU activity on MI300X, not just CPU events. If the image or platform cannot provide GPU profiling, fail the requested profiling run with diagnostics instead of substituting CPU-only profiling. Do not use NVIDIA-only Nsight or vLLM's currently NVIDIA-only Proton integration as an AMD profiler.

## Artifact

Write a unique local `tar.gz` artifact to a user-selectable output directory. It contains:

- A versioned JSON manifest with UTC start/end timestamps, run ID and Pod ID, GPU type/count, data center, image reference/digest if known, model ID/revision if known, vLLM/PyTorch/ROCm/transformers versions, effective serve and benchmark arguments, workload parameters, status, and explicit unavailable values.
- The unmodified benchmark result JSON, baseline metrics snapshots and computed speculative-decoding deltas, and bounded server/worker logs sufficient to diagnose failures.
- When requested and successful, the raw profile traces and summary. Never include API keys, control secrets, model weights, or the Hugging Face cache.

On a successful run the artifact is downloaded and integrity-checked before Pod termination; on timeout, termination takes priority and the artifact may be partial or unavailable. A local JSON sidecar records any artifact checksum and final cleanup outcome afterward. Partial artifacts must be identified as partial, never confused with a successful benchmark. Document how to inspect the JSON and open traces in Perfetto.

## Non-goals

- Multi-GPU or multi-node deployments, automatic cost optimization, production serving, model accuracy evaluation, real-world workload representativeness, or performance guarantees.
- Persistent model caches, retaining Pods by default, or changing the recipe's model, precision, parsers, or speculative-decoding settings.

## Test isolation

All automated tests, including integration and failure-path tests, run without Runpod credentials, GPU hardware, or network access. Use local fakes for the Runpod API, Pod worker, vLLM responses, and time where needed; assert observable lifecycle, recovery, artifact, profiling, and error behavior. CI and normal test commands must never create, query, or delete real Runpod resources. A user's actual benchmark invocation is production use, not a test.

## Acceptance criteria

1. With a funded Runpod account, available MI300X capacity, and network access, one command creates a single-GPU Pod, starts the specified recipe, completes the benchmark, retrieves an intact local artifact, and confirms Pod termination.
2. The artifact records actual configuration, versions, measurements, request failures, and speculative-decoding counters, with unavailable fields clearly identified. Running without `--profile` does not generate a profile.
3. With `--profile`, a separate profiling pass produces an inspectable trace containing ROCm GPU activity; baseline results remain unprofiled.
4. Provisioning, capacity, image compatibility, model download, readiness, benchmark, profiling, download, and cleanup failures return nonzero with actionable diagnostics and preserve retrievable partial results. A failed cleanup identifies the potentially billable Pod.
5. Automated tests exercise lifecycle and security-sensitive failure paths entirely offline, without Runpod credentials or GPU hardware. Report actual Runpod MI300X serving and profiling as unverified unless observed in a real benchmark run; do not perform a live benchmark as a test.
6. If the Pod-scoped key or watchdog is unavailable, the benchmark fails and local cleanup is attempted.
7. Fault-injection tests cover controller `SIGKILL`, host loss, worker never starting, ambiguous create with delayed visibility, failed deletion, pagination, rate limits, and simultaneous invocations. On restart, journaled stray Pods are terminated before creating another Pod; active or unrelated Pods are never deleted. Unconfirmed outcomes remain visible and block further paid runs.

## References

- [Selected vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B.md?hardware=mi300x&variant=fp8&features=tool_calling%2Creasoning%2Cspec_decoding%2Ctext_only)
- [Runpod GPU catalog](https://docs.runpod.io/references/gpu-types) and [HTTP proxy behavior](https://docs.runpod.io/pods/configuration/expose-ports)
- [Runpod Pod termination](https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod), [Pod environment variables](https://docs.runpod.io/pods/templates/environment-variables), and [billing](https://docs.runpod.io/accounts-billing/billing)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) and [profiling](https://docs.vllm.ai/en/stable/contributing/profiling/)
- [Qwen3.8-27B FP8 model](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
