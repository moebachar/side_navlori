"""Collect every result of the Radio Flow SLAM experiments into one JSON for the report page.

  python -m dpro.rf.export_report --runs /root/navlori/runs/dpro --res X:/side_navlori/dpro/rf/results --out report_data.json
"""
import argparse, json, os, pickle
import numpy as np

from ..data import load_golden
from ..evaluate import aligned as align_to
from .graph import fingerprint_distance, propose_pairs
from .revisit import seq_smooth
from .summarize import summarize

MAP_RUNS = ["golden_run_7", "golden_run_10", "golden_run_5", "golden_run_4", "golden_run_9"]
MAP_METHODS = ["v1 network off, tuned", "radio flow + abs, 3-scan average + raw revisits (phase 3 start)",
               "learned update (final refinement)", "ceiling: abs with true APs", "wheel odometry", "odometry + radio flow",
               "odometry + WiFi gyro-bias check", "ceiling: odometry with the true gyro bias removed"]


def rolling(v, k=50):
    v = np.asarray(v, float); out = np.convolve(v, np.ones(k) / k, mode="valid")
    return [round(float(x), 3) for x in out[::10]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    J = lambda p: json.load(open(p)) if os.path.exists(p) else None
    rows = pickle.load(open(f"{args.runs}/rf_results.pkl", "rb"))
    seqs = {s["name"]: s for s in load_golden()}
    rep = dict(summary=summarize(rows), tune=J(f"{args.res}/tune.json"), tune_odo=J(f"{args.res}/tune_odo.json"),
               revisit=J(f"{args.runs}/revisit/revisit_eval.json"), synth=J(f"{args.res}/synth_eval.json"),
               net=J(f"{args.res}/net_behaviour.json"), odo_diag=J(f"{args.res}/odo_diag.json"), gyro=J(f"{args.res}/gyro_bias.json"))
    # phase 2 extras: ceiling at the same number of proposals, availability per run, sequence matching
    sc = dict(np.load(f"{args.runs}/revisit/revisit_eval_scores.npz"))
    avail, orc, seqp = {}, [], {"raw": [], "learned": [], "learned, 3-scan sequence": []}
    for name, s in seqs.items():
        G = s["G"]; N = len(G); GD = np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1])
        avail[name] = int(sum(GD[i, j] < 3 for i in range(N) for j in range(i + 7, N)))
        orc += [GD[i, j] < 3 for i, j, _ in propose_pairs(GD)]
        for lab, M, hib in [("raw", fingerprint_distance(s["R"]), False), ("learned", sc[name], True), ("learned, 3-scan sequence", seq_smooth(sc[name], 3), True)]:
            seqp[lab] += [float(GD[i, j]) for i, j, _ in propose_pairs(M, higher_is_better=hib)]
    rep["revisit_extra"] = dict(available=avail, ceiling_lt3=float(np.mean(orc)),
                                props={k: dict(n=len(v), lt2=float(np.mean(np.array(v) < 2)), lt3=float(np.mean(np.array(v) < 3)), median=float(np.median(v))) for k, v in seqp.items()})
    # phase 3 training curves
    hist = {}
    for s in range(3):
        h = J(f"{args.runs}/rf_net_s{s}/history.json")
        if h:
            hist[s] = dict(pose=rolling([x["pose"] for x in h]), start=rolling([x["start"] for x in h]), flow=rolling([x["flow_db"] for x in h]))
    rep["train"] = hist
    # trajectories for maps (aligned to the truth, seed 0, training 0)
    maps = {}
    for r in MAP_RUNS:
        G = seqs[r]["G"]; maps[r] = dict(G=np.round(G, 2).tolist(), t=np.round(seqs[r]["t"], 1).tolist(), methods={})
        for m in MAP_METHODS:
            cand = [x for x in rows if x["method"] == m and x["run"] == r and x["seed"] == 0 and x.get("tseed", 0) == 0]
            if cand:
                X = np.asarray(cand[0]["X"]); maps[r]["methods"][m] = dict(X=np.round(align_to(X, G), 2).tolist(), aligned=cand[0]["aligned"])
    rep["maps"] = maps
    rep["runs"] = {n: dict(scans=len(s["t"]), seconds=float(s["t"][-1]), path=float(np.sum(np.hypot(*np.diff(s["G"], axis=0).T))), zone=s["zone"]) for n, s in seqs.items()}
    json.dump(rep, open(args.out, "w"))
    print("REPORT_DATA_DONE", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
