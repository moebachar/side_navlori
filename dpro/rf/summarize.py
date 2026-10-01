"""Summarise results.pkl into the report tables (JSON).

Per method and metric: mean over the 12 runs of the per-run value (each run first averaged over evaluation
seeds), with the spread over evaluation seeds and, where there are several trainings, over trainings.
Also per-run values and win counts against the reference solvers.

  python -m dpro.rf.summarize --results results.pkl --out summary.json
"""
import argparse, json, pickle
import numpy as np
import pandas as pd

METRICS = ["aligned", "causal", "online", "drift5"]


def add_causal(rows, seqs=None):
    """causal: the estimate of each scan as it was first made (Xc), aligned like 'aligned' (whole-run rigid fit).
    Unlike 'online' it does not depend on how well the first 30 s fix the map's orientation."""
    from .metrics import aligned
    if any("causal" not in r for r in rows):
        if seqs is None:
            from ..data import load_golden
            seqs = {s["name"]: s for s in load_golden()}
        for r in rows:
            if "causal" not in r:
                r["causal"] = aligned(np.asarray(r["Xc"]), seqs[r["run"]]["G"])
    return rows


def summarize(rows):
    rows = add_causal(rows)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("X", "Xc", "rev", "rev_w", "X_hand")} for r in rows])
    out = {"methods": {}, "per_run": {}}
    for m, g in df.groupby("method"):
        e = {}
        for met in METRICS:
            per_ts = g.groupby(["tseed", "seed"])[met].mean()                  # mean over runs, per (training, seed)
            per_t = per_ts.groupby("tseed").mean()
            e[met] = float(g.groupby("run")[met].mean().mean())
            e[met + "_sd_seed"] = float(per_ts.groupby("tseed").std().mean()) if g.seed.nunique() > 1 else 0.0
            e[met + "_sd_train"] = float(per_t.std()) if len(per_t) > 1 else 0.0
        e["n_rows"] = int(len(g)); e["trainings"] = int(g.tseed.nunique()); e["seeds"] = int(g.seed.nunique())
        out["methods"][m] = e
        out["per_run"][m] = {met: {r: float(v) for r, v in g.groupby("run")[met].mean().items()} for met in METRICS}
    refs = [r for r in ["v1 network off, tuned", "radio flow + abs, 3-scan average + raw revisits (phase 3 start)", "motion rules only"] if r in out["per_run"]]
    out["wins"] = {}
    for m in out["per_run"]:
        out["wins"][m] = {ref: {met: int(sum(out["per_run"][m][met][r] < out["per_run"][ref][met][r] for r in out["per_run"][m][met] if r in out["per_run"][ref][met]))
                                 for met in METRICS} for ref in refs if ref != m}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = pickle.load(open(args.results, "rb"))
    S = summarize(rows)
    json.dump(S, open(args.out, "w"), indent=1)
    order = sorted(S["methods"], key=lambda m: S["methods"][m]["aligned"])
    print(f"{'method':66s} {'aligned':>8s} {'causal':>8s} {'online':>8s} {'drift5':>8s}")
    for m in order:
        e = S["methods"][m]
        print(f"{m:66s} {e['aligned']:8.2f} {e['causal']:8.2f} {e['online']:8.2f} {e['drift5']:8.2f}")


if __name__ == "__main__":
    main()
