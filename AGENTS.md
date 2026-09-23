# AgentX bench

- Enter `.agents/skills/repo-knowledge/SKILL.md` before repository work.
- This is a thin wrapper around pinned official AgentX. Do not reimplement
  replay, filter the corpus, shorten outputs, or silently alter timing.
- `formal` measures 3600 seconds; `smoke` measures 900 seconds after identical
  warmup. Neither profile promises that total wall time equals measurement time.
- Preserve upstream metrics and invalidity reasons. Synthetic payloads do not
  support semantic accuracy or tool-task-success claims.
- The repository does not own model processes or accelerator admission. Never
  start/stop a server or disturb another task's resources as part of client setup.
- Keep credentials, datasets, environments and runtime outputs out of Git.
