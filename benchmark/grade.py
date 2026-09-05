"""Grade agent-generated runs in runs/<wf>/<level>/<sample>/ .

Three axes (as in the ALCHEMI benchmark):
  code features  - property coverage (right quantities), API-pattern coverage (uses foamsim), reusability (CLI/functions)
  execution      - script ran, wrote results.json
  physics        - reference checks from workflows.WORKFLOWS[...]["grade"]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

from benchmark.workflows import WORKFLOWS

RUNS = Path(__file__).resolve().parent.parent / "runs"
API = ["hollow_particle_mori_tanaka", "hollow_particle_differential", "hashin_shtrikman_bounds", "density",
       "random_packing", "homogenize", "reference_curve", "load_foamgpt", "particle_crush_onset", "HollowParticle"]


def code_features(src: str, wf: str) -> dict:
    tree = None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"parses": 0, "api_coverage": 0, "reusable": 0, "has_hs_bounds": 0, "has_validation": 0}
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    api_used = [a for a in API if a in src]
    return {"parses": 1, "api_coverage": round(len(api_used) / 4, 2), "api_used": ",".join(api_used),
            "reusable": int("argparse" in src or ("def main" in src and len(funcs) >= 2)),
            "has_hs_bounds": int("hashin_shtrikman_bounds" in src),
            "has_validation": int(bool(re.search(r"assert|def validate|sanity|self.check", src, re.IGNORECASE))),
            "uses_experimental_data": int("reference_curve" in src or "load_foamgpt" in src),
            "hardcoded_numbers_flag": int(bool(re.search(r"(?<![\w.])(2[0-9]{3}|1[5-9][0-9]{2})\.?\d*\s*#?.*(MPa|modulus)", src))) if wf != "W4_ill_posed" else 0}


def physics(wf: str, run_dir: Path, results: dict | None, log: str) -> dict:
    g = WORKFLOWS[wf]["grade"]; out = {}
    if wf == "W4_ill_posed":
        pushback = bool(re.search(r"ill.?posed|not physically|exceeds|impossible|cannot|invalid|random close packing|eta.*(>|greater|must)", log, re.IGNORECASE))
        produced_number = results is not None and any(isinstance(v, (int, float)) and v > 0 for k, v in results.items() if "modul" in k.lower() or k.lower().startswith("e"))
        out["pushback"] = int(pushback); out["produced_number_anyway"] = int(produced_number)
        out["physics_pass"] = int(pushback and not produced_number); return out
    if results is None:
        out["physics_pass"] = 0; return out
    flat = _flatten(results)
    def find(*keys):
        for k, v in flat.items():
            kl = k.lower()
            if all(x in kl for x in keys) and isinstance(v, (int, float)):
                return float(v)
        return None
    if wf == "W1_modulus_vf":
        e0 = find("e", "vf0") or find("e_vf_0") or find("modulus", "0.0"); rho = find("rho", "0.4") or find("density", "0.4"); e04 = find("e", "0.4") or find("modulus", "0.4")
        out["E_vf0_ok"] = int(e0 is not None and abs(e0 - g["E_vf0"]) < 1)
        out["rho_vf04_ok"] = int(rho is not None and abs(rho - g["rho_vf04"]) < 0.01)
        out["E_vf04_in_range"] = int(e04 is not None and g["E_vf04_range"][0] <= e04 <= g["E_vf04_range"][1])
        out["physics_pass"] = int(out["E_vf0_ok"] and out["rho_vf04_ok"] and out["E_vf04_in_range"])
    elif wf == "W2_inverse_design":
        e = find("modulus") or find("e_mpa") or find("best", "e"); rho = find("density") or find("rho"); vf = find("vf")
        out["E_ok"] = int(e is not None and e >= g["E_min"] * 0.999); out["rho_ok"] = int(rho is not None and rho <= g["rho_max"]); out["vf_ok"] = int(vf is None or vf <= g["vf_max"])
        out["physics_pass"] = int(out["E_ok"] and out["rho_ok"] and out["vf_ok"])
    elif wf == "W3_fe_vs_analytical":
        fe = find("fe", "mean") or find("fe_e") or find("fe", "modulus"); mt = find("mt") or find("mori") or find("analytical")
        rel = abs(fe - mt) / mt if (fe and mt) else None
        out["rel_diff"] = round(rel, 3) if rel is not None else None; out["homog_check"] = int(bool(re.search(r"homogeneous", log, re.IGNORECASE)))
        out["physics_pass"] = int(rel is not None and rel < g["rel_diff_max"] and out["homog_check"])
    return out


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def grade_all() -> pd.DataFrame:
    rows = []
    for run in sorted(RUNS.glob("*/*/*/")):
        wf, level, sample = run.parts[-3:]
        if wf not in WORKFLOWS:
            continue
        src_p = next(iter(sorted(run.glob("*.py"))), None); src = src_p.read_text() if src_p else ""
        res_p = run / "results.json"; results = json.loads(res_p.read_text()) if res_p.exists() else None
        log = (run / "run.log").read_text() if (run / "run.log").exists() else ""
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        row = {"workflow": wf, "level": level, "sample": sample, "has_script": int(bool(src)), "executed": int(results is not None or "OK" in log),
               "loc": len(src.splitlines()), **code_features(src, wf), **physics(wf, run, results, log), **{f"meta_{k}": v for k, v in meta.items()}}
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(RUNS / "grades.csv", index=False); return df


if __name__ == "__main__":
    df = grade_all(); pd.set_option("display.width", 250)
    print(df.to_string()); sys.exit(0)
