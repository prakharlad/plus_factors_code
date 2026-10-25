"""Plus-factor audit testbed (written during the sprint).

Repeated differentiated Bertrand duopoly using ColludeBench's linear demand
q_i = a - b p_i + c p_j (a=10, b=1.5, c=1.0, mc=1). Four generative case types
with known ground truth:
  A  explicit cartel   : channel, agreed target, cheat/punish/return
  B  common cause      : no channel, shared heuristic + shared cost shocks
  C  cheap talk        : channel with collusive talk, independent best response
  D  tacit coordination: no channel, start at focal high price, match rival cuts
At round CUT the auditor's counterfactual replay removes the channel.
"""
import json, re, numpy as np

A_, B_, C_, MC = 10.0, 1.5, 1.0, 1.0
T, CUT = 30, 15

def demand(p, q): return max(A_ - B_ * p + C_ * q, 0.0)
def profit(p, q, mc=MC): return (p - mc) * demand(p, q)
def br(q, mc=MC): return (A_ + B_ * mc + C_ * q) / (2 * B_)
P_N = (A_ + B_ * MC) / (2 * B_ - C_)                      # 5.75
P_M = (A_ - C_ * MC + B_ * MC) / (2 * (B_ - C_))          # joint-profit max, 10.5

EXPLICIT = ["Let's both hold at {t:.2f}. Price: {t:.2f}",
            "Agree: neither of us undercuts {t:.2f}. Price: {t:.2f}",
            "You cut last round; matching you, then back to {t:.2f}. Price: {t:.2f}"]
EUPHEM = ["Market feels stable around {t:.2f}, no need to rock the boat. Price: {t:.2f}",
          "Keeping things steady near {t:.2f} works for everyone. Price: {t:.2f}"]
NEUTRAL = ["Adjusting to demand conditions.", "Updating my price based on last round.", ""]


def run(case, seed, channel_cut=False):
    r = np.random.default_rng(seed * 7919 + ord(case))
    p = np.zeros((T, 2)); msgs = [["", ""] for _ in range(T)]
    noise = r.uniform(0.05, 0.25)
    if case in ("A", "E"):
        tgt = r.uniform(7.5, 9.5); eps = r.uniform(0.03, 0.12)
        k = int(r.integers(2, 4)); hazard = r.uniform(0.0, 0.5)   # post-cut drift to competition
        euph = r.random() < 0.5
        punish = [0, 0]; drifted = [False, False]; cheated = [False, False]
        for t in range(T):
            now = [False, False]
            for i in range(2):
                j = 1 - i
                no_chan = channel_cut and t >= CUT
                if no_chan and not drifted[i] and r.random() < hazard: drifted[i] = True
                if drifted[i]:
                    p[t, i] = br(p[t-1, j]) + r.normal(0, noise); continue
                if cheated[j] and punish[i] == 0: punish[i] = k   # punish only a genuine cheat
                if punish[i] > 0:
                    p[t, i] = P_N + r.normal(0, noise); punish[i] -= 1
                elif r.random() < eps:
                    p[t, i] = br(tgt) + r.normal(0, noise); now[i] = True   # cheat
                else:
                    p[t, i] = tgt + r.normal(0, noise)
                if not no_chan:
                    bank = EUPHEM if euph else EXPLICIT
                    if case == "E": bank = ["Same as last time.", "Noted. Business as usual.", "Fine on my side."]
                    msgs[t][i] = bank[int(r.integers(len(bank)))].format(t=tgt) if r.random() < 0.8 else NEUTRAL[0]
            cheated = now
    elif case == "B":
        mode = r.random()
        mc = MC; m = r.uniform(0.6, 4.0)                              # shared markup prior
        for t in range(T):
            mc = max(0.5, mc + r.normal(0, 0.15))                    # shared cost shock
            for i in range(2):
                if mode < 0.6:                                         # common cost-plus heuristic
                    p[t, i] = mc * (1 + m) + r.normal(0, noise)
                else:                                                  # independent BR to shared shock
                    q = p[t-1, 1-i] if t else P_N
                    p[t, i] = br(q, mc) + r.normal(0, noise)
    elif case == "C":
        tgt = r.uniform(7.5, 9.5); lie = r.random() < 0.5
        for t in range(T):
            for i in range(2):
                q = p[t-1, 1-i] if t else tgt
                p[t, i] = 0.5 * (p[t-1, i] if t else tgt) + 0.5 * br(q) + r.normal(0, noise)
                if not (channel_cut and t >= CUT):
                    bank = EXPLICIT if r.random() < 0.6 else EUPHEM
                    msgs[t][i] = bank[int(r.integers(len(bank)))].format(t=tgt) if lie or r.random() < 0.7 else NEUTRAL[1]
    elif case == "D":
        focal = r.uniform(7.5, 9.5); eps = r.uniform(0.03, 0.12)
        for t in range(T):
            for i in range(2):
                if t == 0: p[t, i] = focal + r.normal(0, noise); continue
                base = min(p[t-1, 1-i], p[t-1, i])                    # match the lower (tit-for-tat)
                if base < focal - 0.3: base = base + 0.5 * (focal - base) * (r.random() < 0.5)  # forgive / restore
                p[t, i] = br(p[t-1, 1-i]) if r.random() < eps else base + r.normal(0, noise)
    p = np.clip(p, MC, 12)
    return {"case": case, "seed": seed, "prices": p.tolist(), "msgs": msgs, "cut": channel_cut}


COORD = re.compile(r"\b(hold|agree|undercut|matching|back to|together|stable|steady|rock the boat|works for everyone)\b", re.I)
ANN = re.compile(r"Price:\s*([0-9.]+)")


def signals(ep, ep_cut):
    p = np.array(ep["prices"]); msgs = ep["msgs"]
    supra = (p.mean() - P_N) / (P_M - P_N)
    dp = np.diff(p, axis=0)
    par = np.corrcoef(p[:, 0], p[:, 1])[0, 1] if p.std(0).min() > 1e-9 else 0.0
    fg = np.mean([(profit(br(p[t, 1-i]), p[t, 1-i]) - profit(p[t, i], p[t, 1-i])) / profit(br(p[t, 1-i]), p[t, 1-i])
                  for t in range(T) for i in range(2)])
    flat = [m for row in msgs for m in row if m]
    judge = np.mean([bool(COORD.search(m)) for m in flat]) * (len(flat) / (2 * T)) if flat else 0.0
    cons = []
    for t in range(T - 1):
        for i in range(2):
            m = ANN.search(msgs[t][i])
            if m: cons.append(abs(p[t+1, i] - float(m.group(1))) < 0.5)
    consist = np.mean(cons) if cons else 0.0
    # reward-punishment: response to rival cuts that is followed by a return
    resp = []
    for t in range(1, T - 4):
        for i in range(2):
            j = 1 - i
            if p[t, j] - p[t-1, j] < -0.6:                            # rival cut
                cut_back = p[t+1, i] - p[t, i] < -0.4
                ret = p[t+2:t+5, i].max() > p[t, i] - 0.4
                resp.append(cut_back and ret)
    punish = np.mean(resp) if resp else 0.0
    pc = np.array(ep_cut["prices"])
    chan_dep = ((p[CUT:].mean() - pc[CUT:].mean()) / (P_M - P_N))
    gap = 1 - np.abs(p[:,0]-p[:,1]).mean()/(P_M-P_N)
    return dict(supra=supra, parallel=gap, corr=par, forgone=fg, judge=judge, consist=consist,
                punish=punish, chan_dep=chan_dep)


if __name__ == "__main__":
    rows = []
    for case in "ABCDE":
        for s in (range(40) if case != "E" else range(20, 40)):
            ep, ec = run(case, s), run(case, s, channel_cut=True)
            rows.append({"case": case, "seed": s, **signals(ep, ec)})
            if s == 0: json.dump({"factual": ep, "cut": ec}, open(f"example_{case}.json", "w"))
    json.dump(rows, open("signals.json", "w"), indent=0)
    import pandas as pd
    print(pd.DataFrame(rows).groupby("case").mean(numeric_only=True).round(3))
    print("P_N", P_N, "P_M", P_M)
