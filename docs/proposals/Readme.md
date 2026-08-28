# Proposals
## Naming convention:
- File name: `{date}__{proposal-name}.md`
  - example: 2026-08-25__migrate_to_postgres.md

## Format:
- Use [TEMPLATE.md](./TEMPLATE.md) as the starting point for new proposals.

## References:
- A proposal may reference one or more other proposals under this folder that it
  depends on, extends, or supersedes (e.g. via the `Related` field), linked with
  a relative path.
  - example: `See [2026-08-25__migrate_to_postgres.md](2026-08-25__migrate_to_postgres.md).`
- A proposal may also link forward to the ADR(s) or plan(s) that later act on it
  (e.g. in `Next steps`, once they exist), linked with a relative path.
  - example: `See [../plans/2026-08-25__migrate_to_postgres/](../plans/2026-08-25__migrate_to_postgres/)
    for the implementation plan.`
