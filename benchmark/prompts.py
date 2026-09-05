"""Print the benchmark prompt for a (workflow, level)."""
import sys

from benchmark.workflows import LEVELS, WORKFLOWS, prompt

if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(prompt(sys.argv[1], sys.argv[2]))
    else:
        for w in WORKFLOWS:
            for lv in LEVELS:
                print(f"\n=== {w} / {lv} ===\n{prompt(w, lv)}")
