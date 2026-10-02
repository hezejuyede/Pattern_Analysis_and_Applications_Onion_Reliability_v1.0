# Portability notes

All executable code discovers the repository root from `__file__`; the release
commands do not require the original Windows workspace path. The historical
absolute command in the analysis run metadata was normalized to the equivalent
repository-relative command for public release. This packaging-only change and
the expansion of split-membership CSV files are documented in
`PUBLIC_REPOSITORY_MIGRATION.md`.
