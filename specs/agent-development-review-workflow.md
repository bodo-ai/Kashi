# Agentic Development Review Pilot

**Status:** Proposed  
**Date:** 2026-11-10  
**Owners:** Engineering

## Intent

Create the repository’s first, deliberately small example of an agentic-development workflow. Developers use a coding agent of their choice; the local guide uses Oh-my-pi (OMP) as its example. GitHub Actions runs independent GitHub Copilot CLI review personas over specifications and pull requests. Roborev reviews local commits only and is never part of CI.

The pilot proves whether AI review findings are useful to humans before we build an agent platform around them.

## Goals

1. Keep specifications in `specs/` as the authoritative, versioned source of implementation intent.
2. Provide a local setup guide for a coding agent, GitHub Copilot, Superpowers, and local-only Roborev.
3. On an explicit `/review` comment, run three independent review lenses for changes to `specs/**`, without repeating completed reviews of individual unchanged spec diffs.
4. On the same request, run general review for changes outside `specs/**` and conformance review when the PR also changes specs.
5. Use organization-billed GitHub Copilot CLI authentication through `GITHUB_TOKEN`; store no developer Copilot credential in CI.
6. Publish each persona’s review as a PR comment, in addition to Actions summaries and artifacts.

## Non-goals

- Automated implementation, test authoring, commits, branches, pull requests, or merges.
- Roborev in GitHub Actions.
- Inline review comments, JSON report schemas, custom report publishers, or a Node/package-tooling project.
- A manifest protocol, separate worktrees, model aliases, LiteLLM, token budgets, fallbacks, or gateways.
- Automated review of fork-originated pull requests.
- Replacing deterministic CI, code ownership, or human PR review.
- Enforcing one local coding agent or installing Roborev’s agent hook by default.

## Decision and considered approaches

| Approach | Decision | Tradeoff |
| --- | --- | --- |
| Hardened review platform | Rejected for the pilot. | Structured reports, manifest isolation, custom publishing, and contract tests provide stronger guarantees, but would create disproportionate setup before review quality is proven. |
| Lean Copilot CLI pilot | Selected. | Five checked-in profiles and one workflow produce human-readable reports. Review completion is required for merge; findings remain advisory. |
| Local-only review | Rejected. | Roborev gives useful commit-level feedback, but it does not provide independent review of specs and pull requests. |

## Repository layout

```text
specs/
  agent-development-review-workflow.md
  README.md
.github/
  agents/
    spec-ambiguity.agent.md
    spec-operability.agent.md
    spec-security.agent.md
    pr-review.agent.md
    conformance.agent.md
  workflows/
    pr-review.yml
.roborev.toml
docs/
  development/
    agentic-workflow.md
```

## Local developer workflow

`docs/development/agentic-workflow.md` must be executable on a clean supported workstation: x64 Linux, including WSL2 Linux, or macOS; each needs Bash, Git, and the current Node.js LTS release. It contains these sections.

### Choose and connect a coding agent

Developers may use any agent compatible with organization policy and repository rules. The OMP example must show how to:

1. install OMP with its current upstream installation command;
2. open OMP in this repository;
3. authenticate OMP with GitHub Copilot through its supported login flow;
4. select an entitled Copilot model; and
5. run a harmless read-only prompt to prove the model connection.

The guide must label OMP-specific instructions as an example, not a requirement.

### Workstation prerequisites

The guide installs the current supported releases of OMP, GitHub Copilot CLI, Superpowers, and Roborev using each project’s upstream Linux instructions. It must state that organization Copilot access is required and that developers authenticate the interactive tools themselves; CI never receives those credentials.

### Install and use Superpowers

Provide the current supported installation method and expected skill-discovery result. State the required workflow: brainstorm and approve a checked-in design before implementation, then produce an implementation plan. Do not instruct developers to bypass an applicable skill.

### Install and configure Roborev locally

The guide must use Roborev’s supported installer and configure the tracked `.roborev.toml` with `agent = "copilot"` and these initial review guidelines: focus on correctness, security, and divergence from an approved specification; do not recommend unrequested validation, retries, abstractions, or style-only changes. It must show `roborev init`, `roborev review`, and `roborev status`; require those guidelines on the default branch; and state that Roborev never runs from `.github/workflows/`.

Roborev’s agent hook is optional and explicitly risk-labeled. The guide must not claim it supports OMP: its current supported-hook list does not include OMP. OMP users retain manual Roborev commands unless support is verified later.

## Copilot CI personas

Profiles live in `.github/agents/*.agent.md`, use `strict-tools-list: true`, and expose only read/search tools. No profile may edit files, invoke GitHub APIs, access the web, run arbitrary shell commands, or request broad permissions.

Every profile must end its output with a short human-readable report containing:

- findings in severity order (`blocker`, `high`, `medium`, `low`, `info`);
- repository path and line range for each finding when applicable;
- rationale and recommended action; and
- an explicit “no findings” result when appropriate.

The workflow—not the model—captures stdout in a text artifact and writes it to the Actions job summary. A separate publisher job posts or updates one PR comment per persona. Findings are advisory; a finding never automatically fails a PR.

| Profile | Scope | Required review question |
| --- | --- | --- |
| `spec-ambiguity` | Changed specification and its explicitly linked repository documents. | Could competent implementers produce materially different behavior? |
| `spec-operability` | Changed specification and related delivery/operational documents. | Does the design name observable failure handling, ownership, and verification where needed? |
| `spec-security` | Changed specification and named security-sensitive interfaces. | Does the design introduce a credible authorization, data, secret, supply-chain, or trust-boundary gap? |
| `pr-review` | Pull-request diff and changed files. | Does the change introduce correctness, regression, security, or repository-rule problems? |
| `conformance` | Pull-request diff and changed specifications. | Does the implementation meet each stated acceptance criterion, or is evidence insufficient? |

## GitHub Actions workflow

The workflow installs the current Copilot CLI, authenticates only through `${{ github.token }}`, and uses `contents: read` and `copilot-requests: write` for review jobs. It runs `copilot --agent <profile> -p <prompt>` noninteractively, using the profile’s strict read-only tool set. It never uses `--allow-all`, `--yolo`, `--allow-all-paths`, `COPILOT_GITHUB_TOKEN`, a PAT, or a personal secret. The publisher has `actions: read`, `pull-requests: write`, and `statuses: write`, but does not execute PR code or run a model.

The organization administrator must enable GitHub Copilot CLI billed to the organization. CI use then draws from the organization’s Copilot entitlement rather than a developer’s credential.

### Requested reviews

`pr-review.yml` runs only on newly created PR conversation comments. An exact `/review` comment from a repository owner, member, or collaborator requests reviews of the current open, non-draft, same-repository PR. For changed files under `specs/**`, it runs `spec-ambiguity`, `spec-operability`, and `spec-security` as independent parallel jobs only for files whose current individual diffs are absent from the bot-authored completed-review record. The record is updated only after all three personas and their publication succeed. Each job supplies the selected spec diffs and file list, uploads its stdout as an artifact, and appends it to its job summary.

### Pull-request review

For changes outside `specs/**`, the request runs `pr-review`. When the same PR also changes specs, it runs `conformance` independently against those changed specs. No governing-spec declaration is required. Each job receives the PR diff, changed-file list, and PR description, then uploads its stdout and writes it to the job summary. Conformance may cite test output when it is present in its inputs, but must label unproven criteria `insufficient evidence`.

Fork-originated PRs are intentionally out of scope for the pilot: they receive no Copilot persona run, artifact, comment, or write-capable token. Human review remains the only review path.

## Spec convention and merge policy

`specs/README.md` defines this repository convention:

- Any file under `specs/` may be a specification; no numbered naming convention or PR declaration is required.
- A specification should be reviewed before implementation starts.

Protect the base branch with the required commit status `agentic-review`, require the branch to be up to date, and apply the rule to administrators. The workflow sets this status on the reviewed PR head and test merge commit to pending, then to success only after every applicable review and publication succeeds against the same head, base, and merge commit; failure or stale results remain blocking. If GitHub has not generated a test merge commit, request another review when it is available. A new head requires another `/review`. Findings remain advisory and do not determine the status. The comment-triggered workflow must first exist on the default branch; bootstrap it before enforcing the required status.

## Acceptance criteria

1. A developer can follow the OMP example to authenticate OMP and Copilot CLI separately to their assigned Copilot seat, run a harmless OMP prompt, run manual local Roborev with `agent = "copilot"`, and inspect the result.
2. The guide supports other coding agents and does not require Roborev’s optional agent hook.
3. An authorized `/review` request reviews only changed specs whose current individual diffs lack a completed review, and exposes each persona report in its job summary, artifact, and PR comment.
4. An authorized `/review` request for a same-repository non-draft PR changing files outside `specs/**` runs general review; when it also changes specs, it runs conformance review against them. Reports appear in summaries, artifacts, and PR comments. Re-runs update each persona’s existing comment.
5. Persona profiles are read-only and the workflow uses neither broad permission bypasses nor personal Copilot credentials.
6. Without an authorized `/review` comment, or for a draft or fork PR, no PR persona runs. The required `agentic-review` status blocks merge until applicable reviews complete.
7. Roborev does not appear in a GitHub Actions workflow.
8. The workflow passes `actionlint`; the local guide is verified on a clean supported workstation; and one same-repository PR verifies the real GitHub Actions surface.

## References

- [GitHub Copilot CLI in GitHub Actions](https://docs.github.com/en/copilot/how-tos/copilot-cli/use-copilot-cli-in-actions)
- [Copilot CLI custom agents](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli)
- [Copilot CLI filesystem policies](https://docs.github.com/en/copilot/concepts/security-governance-and-network-settings/understanding-local-sandboxing)
- [Copilot organization billing](https://docs.github.com/en/copilot/concepts/billing/organizations-and-enterprises)
- [Roborev agent setup](https://roborev.io/docs/agent-setup/)
- [Roborev supported agents](https://roborev.io/docs/agents/)
