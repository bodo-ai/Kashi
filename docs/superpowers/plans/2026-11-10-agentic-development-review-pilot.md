# Agentic Development Review Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lean, advisory GitHub Copilot CLI review pilot for specifications and same-repository pull requests, plus local developer setup guidance.

**Architecture:** Five repository-local Copilot CLI profiles provide independent, read-only review lenses. Two GitHub Actions workflows assemble a temporary review-input file, invoke one profile per job, then publish raw human-readable output only to Actions summaries and artifacts. No application code, custom scripts, schemas, or report-publishing service is introduced.

**Tech Stack:** GitHub Actions, GitHub Copilot CLI, Bash, GitHub Actions artifact and job-summary primitives, Markdown.

**Spec:** `specs/001-agentic-development-review-workflow.md`

## Global Constraints

- Support x64 Linux, WSL2 Linux, and macOS local workstations with Bash, Git, and current Node.js LTS.
- CI uses only `${{ github.token }}` with `contents: read` and `copilot-requests: write`; never add a PAT, `COPILOT_GITHUB_TOKEN`, device OAuth state, or personal secret.
- Every Copilot profile is read-only, uses `strict-tools-list: true`, and receives no web, GitHub API, file-edit, arbitrary-shell, or broad-permission capability.
- Run each persona in a separate Actions job; findings are advisory and must not fail the PR.
- Do not create a custom runner, manifest, report schema, report publisher, Node project, or deterministic CI workflow.
- Never invoke Roborev in CI.
- Run no persona on fork PRs; do not use `pull_request_target`.
- Review output appears only in Actions job summaries and artifacts; do not post PR comments.

## Review Focus

- Fork-originated PRs must skip all persona jobs and never receive `copilot-requests: write`; verify in the actual Actions UI before enabling required checks.
- A profile must not be able to write to the checkout or call external/GitHub tools; exercise each profile once through its workflow on a harmless same-repository PR.
- `spec-review.yml` must run only when `specs/**` changes and must start all three lenses independently; verify this with a spec-only PR.
- PR review must reject drafts and malformed, duplicate, or missing `agentic-review` declarations; verify each eligibility branch with disposable same-repository PRs.
- Conformance must label a criterion `insufficient evidence` when the diff and supplied input do not prove it; use a fixture PR whose specification requires behavior not evidenced by the change.

---

### Task 1: Add local workflow conventions and developer guide

**Files:**
- Create: `.roborev.toml`
- Create: `specs/README.md`
- Create: `docs/development/agentic-workflow.md`

**Interfaces:**
- Produces: `.roborev.toml` with `agent = "copilot"` and the initial review guidelines from the spec.
- Produces: a platform-specific local setup guide for Linux/WSL2 and macOS.
- Produces: repository spec naming and PR-declaration conventions consumed by Tasks 3–4.

- [ ] **Step 1: Create `.roborev.toml`**

Set `agent = "copilot"`. Set `review_guidelines` to focus review on correctness, security, and divergence from approved specifications; reject recommendations for unrequested validation, retries, abstractions, and style-only changes.

- [ ] **Step 2: Write `specs/README.md`**

Define `NNN-kebab-case-title.md`, the spec-first flow, the two allowed PR declaration forms, review-before-implementation, and the written human-exception requirement.

- [ ] **Step 3: Write `docs/development/agentic-workflow.md`**

Use current official installation commands for each supported platform. Include separate Linux/WSL2 and macOS prerequisite/install paths for OMP, Copilot CLI, Superpowers, and Roborev. Show the OMP Copilot login and harmless read-only prompt; state other interactive agents are permitted. Show `roborev init`, `roborev review`, and `roborev status`; state Roborev is local-only. Describe the agent hook as optional, risk-labeled, and unsupported for OMP.

- [ ] **Step 4: Verify the actual local path**

On one supported workstation of each platform family, execute the OMP/Copilot proof prompt, run a manual Roborev review with `agent = "copilot"`, and confirm `roborev status`.

Expected: OMP responds using Copilot; Roborev reports a local review; no setup step creates or changes a GitHub Actions workflow.

- [ ] **Step 5: Commit**

```bash
git add .roborev.toml specs/README.md docs/development/agentic-workflow.md
git commit -m "docs: add local agentic workflow guidance"
```

### Task 2: Define the five read-only Copilot profiles

**Files:**
- Create: `.github/agents/spec-ambiguity.agent.md`
- Create: `.github/agents/spec-operability.agent.md`
- Create: `.github/agents/spec-security.agent.md`
- Create: `.github/agents/pr-review.agent.md`
- Create: `.github/agents/conformance.agent.md`

**Interfaces:**
- Produces: exact profile IDs used by the workflows: `spec-ambiguity`, `spec-operability`, `spec-security`, `pr-review`, and `conformance`.
- Consumes: a workflow-created `review-input.md` file and repository files visible in the read-only checkout.
- Produces: human-readable findings in severity order, with paths/line ranges where applicable, rationale, recommendation, and an explicit no-findings result.

- [ ] **Step 1: Create a shared profile frontmatter convention**

Every profile uses YAML frontmatter with its exact `name`, concise `description`, `strict-tools-list: true`, and only the Copilot CLI read/search tool groups. Do not list edit, execute, web, GitHub, MCP, or subagent tools.

- [ ] **Step 2: Write the three specification-review profiles**

Write one focused prompt per profile:

- ambiguity: materially divergent implementation interpretations;
- operability: missing observable failure handling, ownership, or verification;
- security: credible authorization, data, secret, supply-chain, and trust-boundary risks.

Each instructs the model to review only the changed specification, its explicit links, and `review-input.md`; not to change anything.

- [ ] **Step 3: Write the pull-request and conformance profiles**

`pr-review` reports correctness, regressions, insecure changes, and repository-rule violations from the supplied diff/changed files. `conformance` outputs one row per governing-spec acceptance criterion with `met`, `not met`, or `insufficient evidence`; it may not claim test proof that is absent from `review-input.md`.

- [ ] **Step 4: Smoke-test profiles in an authenticated local Copilot CLI session**

Run: `copilot --agent spec-ambiguity -p "Read review-input.md and report no findings if the fixture is internally consistent." --no-ask-user`

Repeat with each profile and a minimal fixture relevant to its role.

Expected: Each profile loads, produces human-readable output, and cannot offer or perform edits.

- [ ] **Step 5: Commit**

```bash
git add .github/agents
git commit -m "feat: add advisory Copilot review personas"
```

### Task 3: Implement specification-review workflow

**Files:**
- Create: `.github/workflows/spec-review.yml`

**Interfaces:**
- Consumes: the three spec-review profiles from Task 2.
- Produces: three parallel job summaries and three downloadable text artifacts for an eligible same-repository spec PR.

- [ ] **Step 1: Define workflow trigger and safe eligibility guard**

Trigger on `pull_request` types `opened`, `synchronize`, `reopened`, and `ready_for_review` with `paths: ["specs/**"]`. At job level, require `github.event.pull_request.head.repo.fork == false`; do not define a `pull_request_target` trigger.

- [ ] **Step 2: Define the reusable job steps inline in each persona job**

For each profile, use an Ubuntu runner; checkout the PR head with base history sufficient for `git diff`; install the current Copilot CLI; write the base/head spec diff and changed spec-file list to temporary `review-input.md`; invoke `copilot --agent <profile> -p` noninteractively with `GITHUB_TOKEN: ${{ github.token }}` in that step’s environment; save stdout to a text file; append it to `$GITHUB_STEP_SUMMARY`; upload it using `actions/upload-artifact`.

Set job permissions to `contents: read` and `copilot-requests: write`. Do not grant write GitHub permissions, expose secrets, use broad Copilot permission flags, or invoke Roborev.

- [ ] **Step 3: Lint the workflow**

Run: `actionlint .github/workflows/spec-review.yml`

Expected: PASS with no YAML, event, or expression diagnostics.

- [ ] **Step 4: Verify on GitHub Actions**

Open a same-repository PR that changes only a file under `specs/`, then mark it ready for review.

Expected: `spec-ambiguity`, `spec-operability`, and `spec-security` run independently; each creates a readable job summary and text artifact. Open a fork PR changing a spec and confirm the jobs are skipped and receive no Copilot request permission.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/spec-review.yml
git commit -m "feat: review specification changes with Copilot"
```

### Task 4: Implement pull-request review workflow and template

**Files:**
- Create: `.github/pull_request_template.md`
- Create: `.github/workflows/pr-review.yml`

**Interfaces:**
- Consumes: `pr-review` and `conformance` profiles from Task 2 and the governing-spec convention from Task 1.
- Produces: separate advisory general-review and conformance summaries/artifacts for eligible same-repository non-draft PRs.

- [ ] **Step 1: Write the PR template**

Include both exact declaration comments:

```text
<!-- agentic-review: spec=specs/NNN-kebab-case-title.md -->
<!-- agentic-review: spec-exception=human-approved -->
```

Explain that authors retain exactly one valid form, delete the other, and write a non-empty reason when using the exception form.

- [ ] **Step 2: Define eligibility in `pr-review.yml`**

Trigger on the same four `pull_request` event types as Task 3. Use a first, non-persona shell step to exit successfully unless the PR is same-repository, non-draft, and contains exactly one complete allowed declaration with no other `<!-- agentic-review:` marker. For a spec declaration, verify the path matches `specs/[0-9][0-9][0-9]-*.md` and exists at the PR head. For an exception, remove the agentic-review comments and require non-whitespace body text.

- [ ] **Step 3: Define general-review and conformance jobs**

Each job checks out the PR head, writes `review-input.md` containing PR metadata, changed-file list, base/head diff, and governing spec or exception reason. Invoke one profile through `copilot --agent <profile> -p` with `GITHUB_TOKEN: ${{ github.token }}` in that step’s environment, capture stdout, append it to `$GITHUB_STEP_SUMMARY`, and upload it as a text artifact. Give the jobs only `contents: read` and `copilot-requests: write`.

- [ ] **Step 4: Lint and exercise eligibility branches**

Run: `actionlint .github/workflows/pr-review.yml`

Expected: PASS.

Open disposable same-repository PRs covering: draft, fork, missing declaration, duplicate declaration, malformed marker, valid spec declaration, and valid exception with justification.

Expected: Only the two valid same-repository non-draft cases run the personas; every other case exits without a persona invocation.

- [ ] **Step 5: Verify a conformance limitation**

Use a same-repository fixture PR whose governing spec contains an acceptance criterion not proven by the diff or `review-input.md`.

Expected: The conformance artifact labels that row `insufficient evidence`, rather than asserting a passing test or successful behavior.

- [ ] **Step 6: Commit**

```bash
git add .github/pull_request_template.md .github/workflows/pr-review.yml
git commit -m "feat: add advisory pull request review"
```

### Task 5: Run pilot-level verification and enable observation

**Files:**
- Modify: `docs/development/agentic-workflow.md` only if verification exposes an incorrect command or prerequisite.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified Actions workflows and a baseline for the ten-PR signal-quality review in the specification.

- [ ] **Step 1: Lint both workflow files**

Run: `actionlint .github/workflows/spec-review.yml .github/workflows/pr-review.yml`

Expected: PASS.

- [ ] **Step 2: Verify every real Actions surface once**

Complete the same-repository and fork/spec/PR scenarios in Tasks 3–4. Inspect the Actions UI for token permissions, artifacts, summaries, profile isolation, and absence of Roborev.

Expected: The observed behavior matches every acceptance criterion in `specs/001-agentic-development-review-workflow.md`.

- [ ] **Step 3: Record the pilot operating rule in the developer guide**

State that personas are advisory and that engineering evaluates signal quality, false positives, missed defects, workflow duration, and Copilot credit usage after ten same-repository reviewed PRs before adding automation or merge blocking.

- [ ] **Step 4: Commit any verification-driven guide correction**

```bash
git add docs/development/agentic-workflow.md
git commit -m "docs: record review pilot operating rule"
```

Commit only if verification changed the guide.
