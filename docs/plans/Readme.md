# Plans
## Naming convention:
- Single-file plan: `{date}__{plan-name}.md`
  - example: 2026-08-25__update_database.md
- When a plan spans multiple stages, split it into sprints and group them in a
  folder that also carries the date prefix: `{date}__{plan-name}/` (double
  underscore after the date, matching the file convention below)
  - Overview/index file: `{date}__{short-description}.md` (double underscore after
    the date)
    - example: 2026-08-25__telephony_config/2026-08-25__migration_overview.md
  - Sprint file: `{date}__sprint_{NNN}__{sprint-name}.md` — zero-padded 3-digit
    sprint number, double underscore on both sides of `sprint_{NNN}`
    - example: 2026-08-25__telephony_config/2026-08-25__sprint_001__migration.md
    - example: 2026-08-26__tests_for_frontend/2026-08-26__sprint_003__write_tests.md

## Format:
- Use [TEMPLATE.md](./TEMPLATE.md) as the starting point for new plans.

## References:
- A plan may reference one or more proposals under [../proposals](../proposals) and/or
  ADRs under [../adr](../adr) that it implements or is constrained by (e.g. in `Background`),
  linked with a relative path.
  - example: `See [../proposals/2026-08-25__migrate_to_postgres.md](../proposals/2026-08-25__migrate_to_postgres.md)
    and [../adr/2026-08-25__change_db_schema.md](../adr/2026-08-25__change_db_schema.md).`
- A plan may also reference other plans it depends on, blocks, or follows up on, linked
  with a relative path (a sprint file may also reference sibling sprints or its own
  overview file this way).
  - example: `Depends on [../2026-08-20__auth_rework/2026-08-20__migration_overview.md](../2026-08-20__auth_rework/2026-08-20__migration_overview.md).`