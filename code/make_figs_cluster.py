"""Writes the plot data for the cluster figures added for the Cluster Computing version:
  figures/data/lv_{light,heavy,routed}.csv  (Fig. 5: load test vs cluster model)
  figures/data/cx_{policy,hop,split,burst}.csv  (Fig. 8 and Table 6: deployment sensitivity)
Inputs: results/load_validation.csv and the outputs of code/cluster_extra.py.
Run after:  python3 code/cluster_extra.py   (and code/capacity_sensitivity.py for Table 5)
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
D = os.path.join(ROOT, "figures", "data")
os.makedirs(D, exist_ok=True)


def rows(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))


lv = rows("load_validation.csv")
for m in ("light", "heavy", "routed"):
    with open(os.path.join(D, f"lv_{m}.csv"), "w") as f:
        f.write("util,meas,sim\n")
        for r in lv:
            if r["mode"] == m:
                f.write(f"{float(r['load']) * 100:.0f},{float(r['meas_p99_ms']):.3f},{float(r['sim_p99_ms']):.3f}\n")

names = {"central": "Central", "jsq": "JSQ", "rr": "RR", "po2": "Po2", "random": "Random"}
with open(os.path.join(D, "cx_policy.csv"), "w") as f:
    f.write("x,name,light,heavy,routed\n")
    for i, r in enumerate(rows("cx_policy.csv")):
        f.write(f"{i},{names[r['policy']]},{float(r['light_p99_u80']):.2f},"
                f"{float(r['heavy_p99_u80']):.2f},{float(r['routed_p99_u80']):.2f}\n")

for name, key in (("cx_hop", "hop_ms"), ("cx_burst", "scv")):
    with open(os.path.join(D, f"{name}.csv"), "w") as f:
        f.write(f"{key},light,heavy,routed\n")
        for r in rows(f"{name}.csv"):
            vals = [float(r[m + "_cap50"]) for m in ("light", "heavy", "routed")]
            f.write(f"{r[key]}," + ",".join("nan" if v == 0 else f"{v:.1f}" for v in vals) + "\n")

with open(os.path.join(D, "cx_split.csv"), "w") as f:
    f.write("nh,cap200,cap50,sat\n")
    for r in rows("cx_split.csv"):
        f.write(f"{r['n_heavy']},{float(r['cap200']):.1f},{float(r['cap50']):.1f},{float(r['sat']):.1f}\n")
print("figure data written to", D)
