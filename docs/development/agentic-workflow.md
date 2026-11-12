# Local agentic-development workflow

This guide supports x64 Linux (including WSL2 Linux) and macOS with Bash, Git, Node.js LTS, and an organization-assigned GitHub Copilot seat. Developers may use any coding agent. Oh-my-pi (OMP) is the worked example.

CI authenticates separately with GitHub Actions' `GITHUB_TOKEN`. Do not copy a personal Copilot token, OAuth state, or developer credential into repository secrets.

## 1. Install a coding agent and Copilot CLI

### Linux

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
# In OMP: /login github-copilot
```

In OMP, run `/model` and select an entitled `github-copilot/...` model. Then run a harmless read-only prompt such as:

```text
List the files at the repository root. Do not edit anything.
```

Authenticate Copilot CLI separately for local Roborev reviews:

```bash
copilot
# In Copilot CLI: /login
```

## 2. Install Superpowers

Install Superpowers for OMP using its plugin marketplace:

```bash
omp install git:github.com/obra/superpowers
```

Restart OMP and open `/extensions`. Its Skills view should list Superpowers skills such as `brainstorming`. Follow the workflow it enforces: use brainstorming to establish intent, approve a checked-in design before implementation, then produce an implementation plan.

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
roborev wait
roborev show
roborev status
```

Roborev is local-only. No GitHub Actions workflow invokes it.


## 4. PR Reviews

The GitHub Copilot CI personas produce advisory findings, but successful completion of the applicable reviews is required for merge.

Post an exact `/review` as a new PR conversation comment. The PR must be open, non-draft, and from the same repository. Changes under `specs/` receive the three specification reviews, excluding individually unchanged specs whose reviews are already complete. Changes elsewhere receive general review; when the PR also changes specs, conformance review compares the implementation with those specs. No governing-spec declaration is required. Pushes do not request another review; post a new comment for the new head.

