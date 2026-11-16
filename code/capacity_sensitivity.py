"""Sensitivity of the simulated core estimates (Table "design") to modelling assumptions.

Re-derives the minimum number of cores that meets the derated p99 target (results/design_meta.json)
at a given arrival rate for three configurations -- light tier only, routed (theta = 0.9, routed
fraction measured on the development split) and all sentences through the ensemble -- while
perturbing one assumption at a time:
  * service-time variability: every measured sample is multiplied by exp(s*Z - s^2/2), Z ~ N(0,1),
    which keeps the mean and inflates the variance and tail (s = 0 is the measured distribution);
  * dispatcher overhead: one shared FCFS dispatcher with a fixed cost d per dispatch; routed
    requests are dispatched twice (on arrival and when forwarded to the heavy pool);
  * network hop: one-way latency per hop (dispatcher->pool, pool->pool, pool->client);
  * a combined pessimistic case;
  * the derated p99 target itself (a larger or smaller model error at scale).
Numpy only. Output: results/capacity_sensitivity.csv
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_extra as C

RES = C.RES
META = json.load(open(os.path.join(RES, "design_meta.json")))
SLO = META["derated_slo_ms"] / 1000.0
RHO_ROUTED = 0.289          # routed fraction on the development split at theta = 0.9 (design_space.csv)
SEEDS = (0, 1)
N_REQ = 20000


def perturb(samples, s, rng):
    if s <= 0:
        return samples
    return samples * np.exp(s * rng.standard_normal(len(samples)) - s * s / 2)


def simulate(nu, cores, rho, var_s=0.0, disp_s=0.0, hop_s=0.00025, seed=0):
    rng = np.random.default_rng(seed)
    rng_p = np.random.default_rng(1000 + seed)
    sl = perturb(C.SAMPLES["light"], var_s, rng_p)
    sh = perturb(C.SAMPLES["heavy"], var_s, rng_p)
    hop = hop_s + C.PAYLOAD_S
    arr = np.cumsum(rng.exponential(1 / nu, N_REQ))
    t = arr
    if disp_s > 0:
        t = C.stage(t, 1, np.array([disp_s]), "central", rng)
    if rho <= 0:
        done = C.stage(t + hop, cores, sl, "central", rng)
    else:
        nl, nh = C.split_cores(cores, rho)
        d1 = C.stage(t + hop, nl, sl, "central", rng)
        go = rng.random(N_REQ) < rho
        done = d1.copy()
        if go.any():
            t2 = d1[go]
            if disp_s > 0:
                t2 = C.stage(t2, 1, np.array([disp_s]), "central", rng)
            done[go] = C.stage(t2 + hop, nh, sh, "central", rng)
    lat = done + hop - arr
    return lat[N_REQ // 10:]


def ok(nu, cores, rho, slo=None, **kw):
    slo = SLO if slo is None else slo
    if rho > 0:
        nl, nh = C.split_cores(cores, rho)
        if nl < 1 or nh < 1:
            return False
        if nu * C.SAMPLES["light"].mean() >= nl or nu * rho * C.SAMPLES["heavy"].mean() >= nh:
            return False
    elif nu * C.SAMPLES["light"].mean() >= cores:
        return False
    if kw.get("disp_s", 0) * nu * (1 + rho) >= 1:
        return False
    return all(np.percentile(simulate(nu, cores, rho, seed=s, **kw), 99) <= slo for s in SEEDS)


def min_cores(nu, rho, cmax=400, **kw):
    if not ok(nu, cmax, rho, **kw):
        return None
    lo, hi = 1, cmax
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(nu, mid, rho, **kw):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    cases = [("baseline (measured)", {}),
             ("service-time spread s=0.5", {"var_s": 0.5}),
             ("service-time spread s=1.0", {"var_s": 1.0}),
             ("dispatcher 0.1 ms/dispatch", {"disp_s": 0.0001}),
             ("dispatcher 0.25 ms/dispatch", {"disp_s": 0.00025}),
             ("dispatcher 0.4 ms/dispatch", {"disp_s": 0.0004}),
             ("hop 1 ms", {"hop_s": 0.001}),
             ("hop 5 ms", {"hop_s": 0.005}),
             ("combined: s=0.5, 0.25 ms, hop 1 ms", {"var_s": 0.5, "disp_s": 0.00025, "hop_s": 0.001}),
             ("target 150 ms (less derating)", {"slo": 0.150}),
             ("target 75 ms (more derating)", {"slo": 0.075}),
             ("target 50 ms", {"slo": 0.050}),
             ("target 30 ms", {"slo": 0.030})]
    rows = []
    for nu in (1000, 2000):
        for name, kw in cases:
            r = {"nu": nu, "case": name,
                 "light": min_cores(nu, 0.0, **kw),
                 "routed": min_cores(nu, RHO_ROUTED, **kw),
                 "ensemble": min_cores(nu, 1.0, **kw)}
            r["saved_pct"] = (None if not (r["routed"] and r["ensemble"])
                              else round(100 * (1 - r["routed"] / r["ensemble"]), 1))
            rows.append(r); print(r, flush=True)
    with open(os.path.join(RES, "capacity_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
