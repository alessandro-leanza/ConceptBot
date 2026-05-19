# Public / Private Release Split

This repository is the private working tree. Use the export script to generate a sanitized public copy.

## Private Layer

Keep the following private:
- `docs/Reviewer_*.md`, rebuttal drafts, revision insertion notes, and feasibility notes.
- `scripts/experiments/**/results/`, `old_results/`, logs, JSONL traces, plots, and tuning outputs.
- `external_baselines/` vendored repositories and third-party datasets.
- full `instructions/*.json` benchmark files, unless explicitly intended for release.
- `cache/`, `.env`, local model/API caches, and generated artifacts.

## Public Layer

The public export is allowlist-based and currently includes:
- core package files (`README.md`, `LICENSE`, Docker/dependency files);
- core modules in `scripts/modules/` needed to understand/run ConceptBot;
- a minimal instruction example file;
- `scripts/ConceptBot_Main.py` and loader helpers.

## Generate Public Tree

```bash
python scripts/tools/export_public_release.py --output dist/conceptbot-public
```

The output directory is intentionally ignored by git. Review it before pushing:

```bash
find dist/conceptbot-public -maxdepth 3 -type f | sort
```

Then initialize/push that directory as a separate public repository if desired.
