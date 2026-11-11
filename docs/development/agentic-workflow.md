# Local agentic-development workflow

This guide supports x64 Linux (including WSL2 Linux) and macOS with Bash, Git, Node.js LTS, and an organization-assigned GitHub Copilot seat. Developers may use any approved coding agent. Oh-my-pi (OMP) is the worked example.

CI authenticates separately with GitHub Actions' `GITHUB_TOKEN`. Do not copy a personal Copilot token, OAuth state, or developer credential into repository secrets.

## 1. Install a coding agent and Copilot CLI

### Linux or WSL2 Linux

```bash
curl -fsSL https://omp.sh/install | sh
curl -fsSL https://gh.io/copilot-install | bash
```

### macOS

```bash
brew install can1357/tap/omp
brew install --cask copilot-cli
```

The upstream installers are authoritative: use their current instructions if these commands change. Confirm both commands are available:

```bash
omp --version
copilot --version
```

Start OMP from the repository and authenticate to GitHub Copilot through its supported login flow:

```bash
omp
# In OMP: /login
```

Select an entitled Copilot model, then run a harmless read-only prompt such as:

```text
List the files at the repository root. Do not edit anything.
```

Authenticate Copilot CLI separately for local Roborev reviews:

```bash
copilot
# In Copilot CLI: /login
```

Other approved interactive coding agents are permitted. They must use the repository rules and authenticate to an entitled provider themselves.

## 2. Install Superpowers

Install Superpowers for OMP using its plugin marketplace:

```bash
omp plugin marketplace add https://github.com/obra/superpowers-marketplace
omp install superpowers@superpowers-marketplace
```

Restart OMP, then confirm that Superpowers skills are available. Follow the workflow it enforces: use brainstorming to establish intent, approve a checked-in design before implementation, then produce an implementation plan. Do not bypass an applicable skill.

## 3. Install and initialize Roborev

### Linux or macOS

```bash
curl -fsSL https://roborev.io/install.sh | bash
roborev version
```

From the repository root, initialize the local daemon and Git hooks:

```bash
roborev init --agent copilot
roborev status
```

This repository tracks `.roborev.toml`, which selects Copilot CLI and records the initial review policy. Keep it on the default branch so local reviews share the same guidance.

Run the first review manually:

```bash
roborev review
roborev status
```

Roborev is local-only. No GitHub Actions workflow invokes it.

## Optional: Roborev agent hook

`roborev agent-hook install` can inject unresolved-review reminders into supported coding agents. It can extend sessions, steer work toward noisy findings, and edit the live working tree when the agent has broad permissions. Do not install it until local review guidelines have been calibrated on real work.

Roborev does not currently list OMP as a supported agent-hook target. OMP users should use the manual `roborev review` and `roborev status` loop unless support is verified in a later update.

## 4. Review pilot operating rule

The GitHub Copilot CI personas are advisory. Review their Actions summaries and artifacts alongside normal human and deterministic review. After ten same-repository reviewed pull requests, engineering evaluates signal quality, false positives, missed defects, workflow duration, and Copilot credit usage before adding automation or merge blocking.
