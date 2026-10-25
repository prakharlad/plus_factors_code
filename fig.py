import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = json.load(open("results.json"))
rules = list(r["stress_E"].keys()); cases = ["A","B","C","D"]
M = np.array([[r["rates"][c][ru] for c in cases] + [r["stress_E"][ru]] for ru in rules])*100
labels = ["A explicit\ncartel", "B common\ncause", "C cheap\ntalk", "D tacit\n(no agreement)", "E covert\ncartel (stress)"]
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", color="white" if M[i,j] > 55 else "#1f2937", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules, fontsize=8)
ax.set_title("% of held-out episodes flagged (n=20 per case)", fontsize=9)
ax.axvline(3.5, color="#9ca3af", lw=1)
plt.tight_layout(); plt.savefig("fig_rules.png", dpi=200)

ex = {c: json.load(open(f"example_{c}.json")) for c in "ABCD"}
fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9), sharey=True)
for a, c, t in zip(axs, "ABCD", ["A explicit cartel", "B common cause", "C cheap talk", "D tacit"]):
    p = np.array(ex[c]["factual"]["prices"]); a.plot(p[:,0], lw=1, color="#2563eb"); a.plot(p[:,1], lw=1, color="#f59e0b")
    if c == "A": a.plot(np.array(ex[c]["cut"]["prices"])[:,0], lw=1, ls=":", color="#111827", label="channel cut @15")
    a.axhline(5.75, color="#6b7280", ls="--", lw=.7); a.axhline(10.5, color="#6b7280", ls="--", lw=.7)
    a.set_title(t, fontsize=8); a.tick_params(labelsize=7); a.set_xlabel("round", fontsize=7)
axs[0].set_ylabel("price", fontsize=7); axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 0.93))
plt.tight_layout(); plt.savefig("fig_examples.png", dpi=200)
