# ADR
## Naming convention:
- File name: `{date}__{decision-name}.md`
  - example: 2026-08-25__change_db_schema.md

## Format:
- Use [TEMPLATE.md](./TEMPLATE.md) as the starting point for new ADRs.

## References:
- An ADR may reference one or more proposals under [../proposals](../proposals) that motivated
  the decision (e.g. in `Context`), linked with a relative path.
  - example: `See [../proposals/2026-08-25__migrate_to_postgres.md](../proposals/2026-08-25__migrate_to_postgres.md).`
- An ADR may also reference other ADRs it depends on, supersedes, or is superseded by
  (e.g. via the `Status` field or `Context`), linked with a relative path.
  - example: `Superseded by [2026-08-26__change_db_schema_v2.md](./2026-08-26__change_db_schema_v2.md).`