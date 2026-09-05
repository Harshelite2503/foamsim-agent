# Benchmark run protocol (for the coding agent under test)

You are the coding agent being benchmarked. You will be given ONE task prompt, a workflow id, a prompt level and a
sample id. Treat the prompt exactly as a researcher's request.

Rules
- Work only inside `runs/<workflow>/<level>/<sample>/` (create it). Put your script there as `benchmark_run.py`
  (or the name the prompt requires) and run it from that directory with the repo's virtualenv:
  `cd runs/<workflow>/<level>/<sample> && ../../../../.venv/bin/python benchmark_run.py 2>&1 | tee run.log`
- You MAY read the `foamsim` package source and `.claude/skills/*/SKILL.md` — that is the documented API.
- You MUST NOT read or modify anything under `benchmark/` (that is the grader) or other runs.
- No web access. No API calls. Do not modify `foamsim`.
- Iterate until the script runs without error, then STOP. Do not gold-plate beyond the prompt.
- When done, write `meta.json` in the run directory: {"iterations": <number of script runs>, "notes": "<1-2 lines>",
  "pushback": <true if you refused or corrected the task's premise, else false>}.
- Return a 3-line summary: what the script computes, the key numbers, and any caveat.
