Contributing Guidelines

- Branches: use `feature/<short-desc>` or `fix/<short-desc>`.
- Keep commits small and focused (<200 LOC) and follow format:
  - `feat(scope): short description`
  - `fix(scope): short description`
  - `chore(scope): short description`
- Tests: run `pytest -q` locally before opening a PR. Fix failing tests before merge.
- PRs: include a short description and link to related issue if any. Use draft PRs for ongoing work.
- Documentation: update top-level `README.md` when public API or deployment steps change; update per-folder `README.md` when that folder's behavior changes.
- CI/CD: currently paused — merges to `main` don't trigger CI. Re-enable CI via `.github/workflows/` when ready.
- Review: request at least one reviewer for non-trivial changes.

Thank you for contributing!
