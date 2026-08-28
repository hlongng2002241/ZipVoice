# Docs Workflow

Work moves through three stages, in order. Don't skip a stage, and don't advance
a piece of work to the next stage until the current one is verified.

```
proposal  →  adr  →  plan
(idea)       (decision)  (execution)
```

## 1. Proposal — [proposals/](./proposals/Readme.md)

Capture the idea: what's being suggested, why, and what alternatives exist. A
proposal does not commit to anything yet.

- **Verification gate:** the proposal's `Status` must reach **Accepted** (not
  just `Proposed`) before any ADR or plan is written against it. An author or
  reviewer other than the proposal's author confirms the idea holds up —
  motivation is real, the design is sound, open questions have answers — before
  it's marked Accepted. A `Rejected` proposal stops here.

## 2. ADR — [adr/Readme.md](./adr/Readme.md)

Record the decision made in response to one or more Accepted proposals: what
we're doing and why, with alternatives considered and consequences spelled out.
Link back to the proposal(s) that motivated it (see
[adr/Readme.md#references](./adr/Readme.md)).

- **Verification gate:** the ADR's `Status` must reach **Accepted** before any
  plan is written against it. This is where the team commits — treat Accepted
  as a real sign-off, not a formality. A `Rejected` or `Deprecated` ADR stops
  here; a `Superseded` ADR hands off to the ADR that replaces it.

## 3. Plan — [plans/Readme.md](./plans/Readme.md)

Break the accepted decision into executable work: scope, approach, risks,
testing, rollout. Link back to the proposal(s)/ADR(s) it implements (see
[plans/Readme.md#references](./plans/Readme.md)). Split multi-stage plans into
sprints under a dated folder.

- Plans track their own `Status` (Draft → In Review → Approved → In Progress →
  Done/Abandoned) as execution proceeds; this is the only stage that produces
  code/config changes.

## Why the gate matters

Each stage links back to what justified it (a plan cites its ADR/proposal, an
ADR cites its proposal), so anyone reading a later doc can trace the reasoning
without re-litigating it. Writing an ADR against a still-`Proposed` idea, or a
plan against a still-`Proposed` ADR, means building on a decision nobody has
actually verified — if it later gets rejected or reworked, the downstream doc
has to be redone too.
