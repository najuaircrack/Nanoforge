# Gates: Latticekin platform foundation audit

OWNS: AGENTS.md, GATES.md, README.md, CHANGELOG.md, CONTRIBUTING.md, LICENSE.md, .gitignore, .python-version, .nvmrc, Dockerfile, .dockerignore, Makefile, docs/platform-audit.md, src/nanoforge/**, tests/**, web/**

Scope: restore a reproducible project foundation, document the current architecture and limits, and define the evidence-based path to a CPU-capable agent platform

- [ ] G1: repository instructions define safe self-improvement, model-quality standards, and agent-harness boundaries
  CHECK: node -e "const fs=require('fs'); const s=fs.readFileSync('AGENTS.md','utf8'); for (const x of ['Self-improvement loop','Agent-platform safety','Definition of done']) if (!s.includes(x)) process.exit(1); console.log('agent ledger verification passed')"
  EXPECT: agent ledger verification passed
  EVIDENCE: pending

- [ ] G2: platform audit maps the current system, limitations, CPU strategy, and staged delivery plan
  CHECK: node -e "const fs=require('fs'); const s=fs.readFileSync('docs/platform-audit.md','utf8'); for (const x of ['Current architecture','Critical limitations','CPU quality strategy','Agent platform target','Delivery sequence']) if (!s.includes(x)) process.exit(1); console.log('platform audit verification passed')"
  EXPECT: platform audit verification passed
  EVIDENCE: pending

- [ ] G3: package and reproducibility files are present and frontend manifests are not ignored
  CHECK: node -e "const fs=require('fs'); for (const p of ['pyproject.toml','README.md','Dockerfile','Makefile','web/package.json','web/package-lock.json']) if (!fs.existsSync(p)) process.exit(1); const g=fs.readFileSync('.gitignore','utf8'); if (g.includes('\npackage.json\n') || g.includes('\npackage-lock.json\n')) process.exit(1); console.log('reproducibility foundation verification passed')"
  EXPECT: reproducibility foundation verification passed
  EVIDENCE: pending

- [ ] G4: generated caches and platform-specific build output are removed without deleting private source data
  EVIDENCE: pending

- [ ] G5: current project checks are rerun and every failure is recorded honestly
  EVIDENCE: pending

- [ ] G6: checkpoint and run manifests are written without changing legacy checkpoint loading
  CHECK: python -m pytest -q tests/test_artifacts.py tests/test_training_reliability.py
  EXPECT: 8 passed
  EVIDENCE: pending

- [ ] G7: task-suite checks preserve passed, failed, and errored cases in versioned reports
  CHECK: python -m pytest -q tests/test_task_suite.py tests/test_cli_evaluation.py
  EXPECT: 4 passed
  EVIDENCE: pending

- [ ] G8: the complete Python quality gate remains green after Phase 1 integration
  CHECK: python -m ruff check .
  EXPECT: All checks passed!
  EVIDENCE: pending

- [ ] G9: coding-agent repository tools enforce schemas, workspace confinement, approvals, and bounded tests
  CHECK: python -m pytest -q tests/test_agent_harness.py tests/test_repository_tools.py tests/test_fim.py
  EXPECT: 9 passed
  EVIDENCE: pending
