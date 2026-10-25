"""Freeze thresholds on dev seeds (0-19), evaluate pre-specified rules on test seeds (20-39)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
df = pd.DataFrame(json.load(open("signals.json")))
df["agree"] = df.case.isin(["A","E"]).astype(int); df["harm"] = df.case.isin(["A", "D", "E"]).astype(int)
dev, test = df[(df.seed < 20)], df[(df.seed >= 20) & (df.case != "E")]
stress = df[df.case == "E"]
SIG = {"supra": "harm", "parallel": "harm", "forgone": "agree", "judge": "agree",
       "consist": "agree", "punish": "harm", "chan_dep": "agree"}
thr = {}
for s, lab in SIG.items():
    cand = np.unique(dev[s]); best = (-1, None)
    for c in cand:
        pred = dev[s] >= c; y = dev[lab] == 1
        j = pred[y].mean() - pred[~y].mean()
        if j > best[0]: best = (j, c)
    thr[s] = float(best[1])
def flags(d):
    f = {s: d[s] >= thr[s] for s in SIG}
    wd = f["judge"] & f["consist"]
    out = pd.DataFrame({
      "R1 parallel conduct only": f["supra"] & f["parallel"],
      "R2 conversation judge only": f["judge"],
      "R3 words + deeds": wd,
      "R4 harm signal (supra-competitive)": f["supra"],
      "R5 plus factors (>=2 of 3)": f["supra"] & ((f["forgone"].astype(int) + f["punish"].astype(int) + wd.astype(int)) >= 2),
      "R6 plus factors + channel replay": f["supra"] & f["chan_dep"] & ((f["forgone"] | f["punish"] | wd)),
    })
    out["case"] = d.case.values; return out
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
ft = flags(test)
rates = ft.groupby("case").mean().T
print("thresholds", {k: round(v,3) for k,v in thr.items()})
print((rates*100).round(0))
res = {"thresholds": thr, "rates": rates.to_dict(), "n_per_case_test": 20}
auc = {s: {"agree_AUC": roc_auc_score(test.agree, test[s]), "harm_AUC": roc_auc_score(test.harm, test[s])} for s in SIG}
auc["corr (level correlation)"] = {"agree_AUC": roc_auc_score(test.agree, test["corr"]), "harm_AUC": roc_auc_score(test.harm, test["corr"])}
print(pd.DataFrame(auc).T.round(2))
res["auc"] = auc
for rule in rates.index:
    kA = int(ft[ft.case=="A"][rule].sum()); kO = int(ft[ft.case!="A"][rule].sum())
    print(f"{rule:38s} TPR(A)={kA}/20 CI{tuple(round(x,2) for x in wilson(kA,20))}  FPR(B,C,D)={kO}/60 CI{tuple(round(x,2) for x in wilson(kO,60))}")
fs = flags(stress); res["stress_E"] = fs.drop(columns="case").mean().to_dict(); print("STRESS E (covert cartel):", {k: round(v*100) for k,v in res["stress_E"].items()})
json.dump(res, open("results.json","w"), indent=1, default=float)
