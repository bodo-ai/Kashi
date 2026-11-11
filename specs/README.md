# Specifications

Specifications are the repository’s authoritative, versioned implementation intent.

## Naming

Name each specification `NNN-kebab-case-title.md`, where `NNN` is the next three-digit sequence number.

## Flow

1. Draft and approve a specification in `specs/` before implementation begins.
2. Open a specification-only pull request when the design changes.
3. Implement against the approved specification.
4. Name the governing specification in every feature pull request.

## Pull-request declaration

A feature pull request must contain exactly one of these comments in its body:

```text
<!-- agentic-review: spec=specs/NNN-kebab-case-title.md -->
<!-- agentic-review: spec-exception=human-approved -->
```

Keep the declaration that applies and delete the other one. A `spec` declaration must reference an existing file in `specs/`. A `spec-exception` declaration requires a non-empty, human-written explanation outside the comment.

The review pilot treats missing, malformed, or duplicate `agentic-review` declarations as ineligible for pull-request personas. A human may approve an exception, but the reason remains part of the pull-request record.
