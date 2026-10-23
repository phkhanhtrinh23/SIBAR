"""Experiments of Section 5.

  python scripts/run.py prepare  --dataset crosswoz
  python scripts/run.py runs     --dataset crosswoz            # Table 4
  python scripts/run.py main     --dataset crosswoz --replays 40 --workers 16   # Table 5, Fig. 3
  python scripts/run.py reprice  --dataset crosswoz            # Table 6 (re-prices main results)
  python scripts/run.py ablation --dataset crosswoz            # Fig. 4b
  python scripts/run.py sweep    --dataset crosswoz --param horizon|budget|oracle_mix|gamma   # Fig. 5
  python scripts/run.py churn    --dataset crosswoz            # Fig. 6
  python scripts/run.py tune     --dataset crosswoz [--param budget|churn|horizon]   # on the dev split
  python scripts/run.py latency  --dataset crosswoz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sibar.analysis import run_statistics  # noqa: E402
from sibar.config import SibarConfig  # noqa: E402
from sibar.experiment import ALL_METHODS, Runner, load_dataset, load_results, save_results  # noqa: E402
from sibar.metrics import format_table, position_avoid, summarize  # noqa: E402

TUNE_H, TUNE_GAMMA = (2, 3, 4, 6), (0.5, 0.7, 0.9, 1.0)
SWEEP_GRID = {"horizon": [1, 2, 3, 4, 6], "budget": [32, 64, 128, 256], "gamma": [0.5, 0.7, 0.9, 1.0],
              "oracle_mix": [0.0, 0.25, 0.5, 0.75, 1.0]}
CHURN_RATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def out_dir(args) -> str:
    d = os.path.join(args.out, args.dataset)
    os.makedirs(d, exist_ok=True)
    return d


def sessions(args, ds, split="test"):
    ss = getattr(ds, split)
    if args.max_sessions and len(ss) > args.max_sessions:
        idx = np.random.default_rng(0).choice(len(ss), args.max_sessions, replace=False)
        ss = [ss[i] for i in sorted(idx)]
    return ss


def make_runner(args, ds, cfg):
    return Runner(ds, cfg, cache_dir=args.cache, encoder=args.encoder)


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    print(f"[saved] {path}")


def cmd_prepare(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed, rebuild=True)
    print(json.dumps(ds.stats(), indent=2))


def cmd_runs(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    st = run_statistics(ds.train + ds.dev + ds.usability_fold, ds.test)
    for k, v in st.items():
        print(f"{k:<58}{v:8.1f}")
    dump(os.path.join(out_dir(args), "table4_runs.json"), st)


def cmd_main(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    methods = args.methods.split(",") if args.methods else ALL_METHODS
    if "Reply-Only" not in methods:
        methods = ["Reply-Only"] + methods
    res = runner.run(methods, sessions(args, ds), args.replays, args.workers, cfg_for=tuned(args, cfg),
                     desc="main")
    save_results(res, os.path.join(out_dir(args), "main.jsonl"))
    summ = summarize(res, cfg, reference="Sibar" if "Sibar" in methods else None)
    print(format_table(summ, methods))
    pos = position_avoid(res)
    print("\nAvoid (%) at the 1st / 2nd follow-up after a search reply (Fig. 3)")
    for m in methods:
        if m in pos:
            print(f"  {m:<22}{pos[m].get(1, 0):6.1f} / {pos[m].get(2, 0):6.1f}")
    for ref in ("One-Step", "Reply-Aware"):
        if ref in methods and "Sibar" in methods:
            d = summarize(res, cfg, reference=ref)["Sibar"]["diff"]
            print(f"Sibar - {ref}: Avoid {d['Avoid'][0]:+.1f} [{d['Avoid'][1]:.1f}, {d['Avoid'][2]:.1f}]"
                  f"  Net {d['Net'][0]:+.1f} [{d['Net'][1]:.1f}, {d['Net'][2]:.1f}]")
    dump(os.path.join(out_dir(args), "table5_main.json"), {"summary": summ, "positions": pos})


def cmd_reprice(args, cfg):
    res = load_results(os.path.join(out_dir(args), "main.jsonl"))
    rows = {}
    settings = [("lam", v) for v in (0.0003, 0.0006, 0.0012, 0.0024)] + [("c_err", 1.0), ("c_err", 4.0)] + \
               [("c_pred", 0.01), ("c_pred", 0.10)]
    for k, v in settings:
        c = cfg.replace(**{k: v})
        s = summarize(res, c, reference="Reply-Aware")
        rows[f"{k}={v}"] = {m: s[m]["point"]["Net"] for m in ("One-Step", "Reply-Aware", "Sibar") if m in s}
        if "Sibar" in s:
            rows[f"{k}={v}"]["interval Sibar-Reply-Aware"] = s["Sibar"]["diff"]["Net"][1:]
        print(k, v, rows[f"{k}={v}"])
    c = cfg.replace(charge_pred_every_reply=True)
    s = summarize(res, c)
    rows["C_pred at every reply"] = {m: s[m]["point"]["Net"] for m in s}
    print("C_pred at every reply", rows["C_pred at every reply"])
    dump(os.path.join(out_dir(args), "table6_reprice.json"), rows)


def cmd_ablation(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    cfg = tuned(args, cfg)["Sibar"]
    variants = [("Reply-Only", "Reply-Only", cfg), ("Sibar", "Sibar", cfg),
                ("Greedy selector", "Sibar", cfg.replace(selector="greedy")),
                ("w/o cache digest", "Sibar", cfg.replace(use_digest=False)),
                ("w/o cross-user scen.", "Sibar", cfg.replace(scenario_source="session_only")),
                ("w/o backoff", "Sibar", cfg.replace(use_backoff=False)),
                ("w/o complete-evidence gate", "Sibar", cfg.replace(complete_evidence_gate=False))]
    res = runner.run(variants, sessions(args, ds), args.replays, args.workers, desc="ablation")
    s = summarize(res, cfg, reference="Sibar")
    print(format_table(s, [v[0] for v in variants]))
    dump(os.path.join(out_dir(args), "fig4b_ablation.json"), s)


def cmd_sweep(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    out = {}
    for v in SWEEP_GRID[args.param]:
        # budget and horizon use configurations re-tuned on dev for each value (Appendix B)
        t = tuned_at(args, cfg, args.param, v) if args.param in ("budget", "horizon") else tuned(args, cfg)
        c = cfg.replace(**{args.param: v})
        sib, ra = t["Sibar"].replace(**{args.param: v}), t["Reply-Aware"].replace(**{args.param: v})
        methods = [("Reply-Only", "Reply-Only", c), ("Sibar", "Sibar", sib), ("Reply-Aware", "Reply-Aware", ra)]
        if args.param != "horizon":
            methods.append(("One-Step", "One-Step", c))
        res = runner.run(methods, sessions(args, ds), args.replays, args.workers, desc=f"{args.param}={v}")
        s = summarize(res, c, reference="Sibar")
        out[str(v)] = {m: {"Net": s[m]["point"]["Net"], "Avoid": s[m]["point"]["Avoid"],
                           "diff_vs_Sibar": s[m].get("diff", {}).get("Net")} for m in s}
        print(args.param, v, {m: round(out[str(v)][m]["Net"], 2) for m in out[str(v)]})
    dump(os.path.join(out_dir(args), f"fig5_{args.param}.json"), out)


def cmd_churn(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    out = {}
    for rate in CHURN_RATES:
        ds.catalog.set_churn(rate)
        t = tuned_at(args, cfg, "churn", rate)
        methods = [("Reply-Only", "Reply-Only", cfg), ("Sibar", "Sibar", t["Sibar"]),
                   ("Lease-blind Sibar", "Sibar", t["Sibar"].replace(lease_aware=False)),
                   ("Reply-Aware", "Reply-Aware", t["Reply-Aware"]), ("One-Step", "One-Step", cfg)]
        res = runner.run(methods, sessions(args, ds), args.replays, args.workers, desc=f"churn={rate}")
        s = summarize(res, cfg, reference="Sibar")
        out[str(rate)] = {m: {"Unsup": s[m]["point"]["Unsup"], "Net": s[m]["point"]["Net"],
                              "diff_vs_Sibar": s[m].get("diff", {}).get("Net")} for m in s}
        print(rate, {m: (round(v["Unsup"], 2), round(v["Net"], 2)) for m, v in out[str(rate)].items()})
    ds.catalog.set_churn(0.0)
    dump(os.path.join(out_dir(args), "fig6_churn.json"), out)


def tune_on_dev(args, runner, dev, cfg, desc, horizons=TUNE_H, reply_aware=True) -> dict:
    """Appendix B: each policy is tuned for Net with at most 16 configurations on the development
    split. Sibar tries H x gamma (16), Reply-Aware the same H (4)."""
    grid = [("Reply-Only", "Reply-Only", cfg)]
    for H in horizons:
        for g in TUNE_GAMMA:
            grid.append((f"Sibar H={H} g={g}", "Sibar", cfg.replace(horizon=H, gamma=g)))
        if reply_aware:
            grid.append((f"Reply-Aware H={H}", "Reply-Aware", cfg.replace(horizon=H)))
    res = runner.run(grid, dev, args.replays, args.workers, desc=desc)
    s = summarize(res, cfg, n_boot=10)
    net = {m: s[m]["point"]["Net"] for m in s}
    best = {"Sibar": max((m for m in s if m.startswith("Sibar")), key=net.get)}
    if reply_aware:
        best["Reply-Aware"] = max((m for m in s if m.startswith("Reply-Aware")), key=net.get)
    out = {}
    for pol, m in best.items():
        kv = dict(x.split("=") for x in m.split()[1:])
        out[pol] = {"horizon": int(kv["H"])} | ({"gamma": float(kv["g"])} if "g" in kv else {})
    print(f"[{desc}] best: " + "; ".join(best.values()))
    return {"best": out, "dev_net": net}


def cmd_tune(args, cfg):
    """--param default: H and gamma for the main results; budget / churn: re-tuned for each value
    (Fig. 5b, Fig. 6); horizon: gamma selected for each H (Fig. 5a)."""
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    dev = sessions(args, ds, "dev")
    if args.param == "default":
        dump(os.path.join(out_dir(args), "tune.json"), tune_on_dev(args, runner, dev, cfg, "tune"))
        return
    out = {}
    if args.param == "budget":
        for b in SWEEP_GRID["budget"]:
            out[str(b)] = tune_on_dev(args, runner, dev, cfg.replace(budget=b), f"tune budget={b}")
    elif args.param == "churn":
        for rate in CHURN_RATES:
            ds.catalog.set_churn(rate)
            out[str(rate)] = tune_on_dev(args, runner, dev, cfg, f"tune churn={rate}")
        ds.catalog.set_churn(0.0)
    elif args.param == "horizon":
        for H in SWEEP_GRID["horizon"]:
            r = tune_on_dev(args, runner, dev, cfg, f"tune gamma at H={H}", horizons=(H,), reply_aware=False)
            r["best"]["Reply-Aware"] = {"horizon": H}
            out[str(H)] = r
    else:
        raise ValueError(f"tune --param {args.param}: use default | budget | churn | horizon")
    dump(os.path.join(out_dir(args), f"tune_{args.param}.json"), out)


def tuned(args, cfg, best: dict = None) -> dict:
    """Per-method configurations fixed on dev (outputs/<dataset>/tune.json); falls back to the
    config file when the dataset has not been tuned."""
    if best is None:
        path = os.path.join(out_dir(args), "tune.json")
        if os.path.exists(path):
            with open(path) as f:
                best = json.load(f)["best"]
        else:
            print(f"[warn] {path} missing: using H={cfg.horizon}, gamma={cfg.gamma} from the config")
            best = {}
    sib = cfg.replace(**best.get("Sibar", {}))
    ra = cfg.replace(**best.get("Reply-Aware", {}))
    return {"Sibar": sib, "Hindsight Sibar": sib, "Reply-Aware": ra}


def tuned_at(args, cfg, param: str, value) -> dict:
    """Configurations re-tuned on dev for one sweep value (outputs/<dataset>/tune_<param>.json)."""
    path = os.path.join(out_dir(args), f"tune_{param}.json")
    if not os.path.exists(path):
        print(f"[warn] {path} missing: {param}={value} uses the main configuration")
        return tuned(args, cfg)
    with open(path) as f:
        return tuned(args, cfg, json.load(f)[str(value)]["best"])


def cmd_latency(args, cfg):
    ds = load_dataset(args.dataset, args.raw, args.processed)
    runner = make_runner(args, ds, cfg)
    ss = sessions(args, ds)[:200]
    cfg = tuned(args, cfg)["Sibar"]
    for H in (1, cfg.horizon):
        lat = runner.latencies(cfg.replace(horizon=H), ss)
        print(f"H={H}: median {1000 * np.median(lat):.1f} ms per reply, p90 {1000 * np.percentile(lat, 90):.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "runs", "main", "reprice", "ablation", "sweep", "churn",
                                        "tune", "latency"])
    ap.add_argument("--dataset", default="crosswoz", choices=["crosswoz", "multiwoz", "synthetic", "vogue"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--replays", type=int, default=1, help="replays per session (paper: 40)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-sessions", type=int, default=0, help="subsample test sessions (0 = all)")
    ap.add_argument("--methods", default="", help="comma-separated subset of methods for `main`")
    ap.add_argument("--param", default="default",
                    help="sweep: horizon|budget|gamma|oracle_mix; tune: default|budget|churn|horizon")
    ap.add_argument("--encoder", default=None, help="override the state encoder ('hash' runs offline)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default=".cache")
    args = ap.parse_args()
    cfg = SibarConfig.from_yaml(args.config) if os.path.exists(args.config) else SibarConfig()
    globals()[f"cmd_{args.command}"](args, cfg)


if __name__ == "__main__":
    main()
