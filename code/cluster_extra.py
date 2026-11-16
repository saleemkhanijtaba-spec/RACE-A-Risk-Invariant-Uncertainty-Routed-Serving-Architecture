"""Cluster-level sensitivity analyses of the routed serving architecture (simulation).

Uses the same measured per-request service times (results/latency_samples.csv), routed
fraction (results/sim_meta.json) and network-hop model as simulate.py, and extends the
discrete-event model with
  (1) dispatch policies: central FCFS queue per pool, join-shortest-queue (JSQ),
      power-of-two choices, round robin and random assignment to per-worker queues;
  (2) the one-way network hop latency, which grows when tiers sit on different nodes;
  (3) the split of a fixed core budget between the light and heavy pools;
  (4) bursty arrivals (hyper-exponential inter-arrival times with a given squared
      coefficient of variation, SCV).
None of these knobs changes which model serves which sentence, so none of them changes
the false-edit rate; only latency and capacity are reported.
Outputs: results/cx_policy.csv, cx_hop.csv, cx_split.csv, cx_burst.csv, cx_meta.json
Runs on numpy only (no PyTorch).
"""
import os, csv, json, heapq, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SLO_S = 0.200
BW_BPS = 10e9 / 8
N_REQ = 20000
SEEDS = (0, 1)
CORES = 16


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


lat = read("latency_samples.csv")
SAMPLES = {t: np.array([float(r["ms"]) / 1000 for r in lat if r["tier"] == t]) for t in ("light", "heavy")}
META = json.load(open(os.path.join(RES, "sim_meta.json")))
RHO = META["rho"]
PAY = json.load(open(os.path.join(RES, "payload.json")))
PAYLOAD_S = (PAY["req_bytes_mean"] + PAY["resp_bytes_mean"]) / BW_BPS


def arrivals(lam, n, scv, rng):
    if scv <= 1.0 + 1e-9:
        return np.cumsum(rng.exponential(1 / lam, n))
    # balanced-means two-phase hyper-exponential with mean 1/lam and the given SCV
    p = 0.5 * (1 + np.sqrt((scv - 1) / (scv + 1)))
    r1, r2 = 2 * p * lam, 2 * (1 - p) * lam
    pick = rng.random(n) < p
    gaps = np.where(pick, rng.exponential(1 / r1, n), rng.exponential(1 / r2, n))
    return np.cumsum(gaps)


def stage(t_in, k, samples, policy, rng):
    """Serve jobs arriving at times t_in (any order) on k single-core workers; returns finish times."""
    order = np.argsort(t_in, kind="stable")
    done = np.empty(len(t_in))
    svc = rng.choice(samples, len(t_in))
    if policy == "central":
        free = [0.0] * k
        heapq.heapify(free)
        for i in order:
            f = heapq.heappop(free)
            e = max(t_in[i], f) + svc[i]
            heapq.heappush(free, e)
            done[i] = e
        return done
    q = [collections.deque() for _ in range(k)]      # outstanding finish times per worker
    last = [0.0] * k
    rr = 0
    for i in order:
        t = t_in[i]
        for d in q:
            while d and d[0] <= t:
                d.popleft()
        if policy == "jsq":
            m = min(len(d) for d in q)
            cand = [w for w in range(k) if len(q[w]) == m]
            w = cand[rng.integers(len(cand))]
        elif policy == "po2":
            a, b = rng.choice(k, 2, replace=False)
            w = a if len(q[a]) <= len(q[b]) else b
        elif policy == "rr":
            w = rr; rr = (rr + 1) % k
        elif policy == "random":
            w = int(rng.integers(k))
        else:
            raise ValueError(policy)
        e = max(t, last[w]) + svc[i]
        last[w] = e
        q[w].append(e)
        done[i] = e
    return done


def simulate(lam, mode, n=CORES, split=None, policy="central", hop_s=0.00025, scv=1.0, seed=0, n_req=N_REQ):
    rng = np.random.default_rng(seed)
    hop = hop_s + PAYLOAD_S
    arr = arrivals(lam, n_req, scv, rng)
    if mode == "light":
        done = stage(arr + hop, n, SAMPLES["light"], policy, rng)
    elif mode == "heavy":
        done = stage(arr + hop, n, SAMPLES["heavy"], policy, rng)
    else:
        nl, nh = split if split else split_cores(n)
        d1 = stage(arr + hop, nl, SAMPLES["light"], policy, rng)
        go = rng.random(n_req) < RHO
        done = d1.copy()
        if go.any():
            done[go] = stage(d1[go] + hop, nh, SAMPLES["heavy"], policy, rng)
    lats = done + hop - arr
    return lats[n_req // 10:]


def split_cores(n, rho=RHO):
    wl = SAMPLES["light"].mean(); wh = rho * SAMPLES["heavy"].mean()
    nh = min(max(1, int(round(n * wh / (wl + wh)))), n - 1)
    return n - nh, nh


def ok(lam, slo=SLO_S, **kw):
    return all(np.percentile(simulate(lam, seed=s, **kw), 99) <= slo for s in SEEDS)


def saturation(mode, n=CORES, split=None):
    """Arrival rate at which the busiest pool reaches 100% utilisation."""
    if mode == "routed":
        nl, nh = split if split else split_cores(n)
        return min(nl / SAMPLES["light"].mean(), nh / (RHO * SAMPLES["heavy"].mean()))
    return n / SAMPLES[mode].mean()


def p99_at(util, mode, n=CORES, split=None, **kw):
    lam = util * saturation(mode, n, split)
    return 1000 * float(np.mean([np.percentile(simulate(lam, mode=mode, n=n, split=split, seed=s, **kw), 99) for s in SEEDS]))


def capacity(mode, n=CORES, split=None, slo=SLO_S, **kw):
    hi = saturation(mode, n, split)
    lo = 0.0
    for _ in range(13):
        mid = (lo + hi) / 2
        if ok(mid, slo=slo, mode=mode, n=n, split=split, **kw):
            lo = mid
        else:
            hi = mid
    return lo


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    modes = ("light", "heavy", "routed")
    sat = {m: saturation(m) for m in modes}

    def row(r, **kw):
        for m in modes:
            r[m + "_p99_u80"] = p99_at(0.8, m, **kw)
            r[m + "_cap200"] = capacity(m, slo=0.200, **kw)
            r[m + "_cap50"] = capacity(m, slo=0.050, **kw)
        print(r, flush=True)
        return r

    write("cx_policy.csv", [row({"policy": p}, policy=p) for p in ("central", "jsq", "po2", "rr", "random")])
    write("cx_hop.csv", [row({"hop_ms": h}, hop_s=h / 1000) for h in (0.05, 0.25, 0.5, 1, 2, 5, 10, 20)])
    write("cx_burst.csv", [row({"scv": v}, scv=v) for v in (1, 2, 4, 8, 16)])
    rows = []
    prop = split_cores(CORES)
    for nh in range(2, CORES - 1):
        sp = (CORES - nh, nh)
        r = {"n_light": CORES - nh, "n_heavy": nh, "proportional": int(nh == prop[1]),
             "sat": saturation("routed", split=sp), "cap200": capacity("routed", split=sp, slo=0.200),
             "cap50": capacity("routed", split=sp, slo=0.050)}
        rows.append(r); print(r, flush=True)
    write("cx_split.csv", rows)
    json.dump({"cores": CORES, "rho": RHO, "split_prop": prop, "n_req": N_REQ, "seeds": SEEDS,
               "payload_s": PAYLOAD_S, "saturation": sat,
               "mean_ms": {k: 1000 * float(v.mean()) for k, v in SAMPLES.items()}},
              open(os.path.join(RES, "cx_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
