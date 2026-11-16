"""
rf_engine.py
Mesin Random Forest (C4.5 / Gain Ratio) yang dibuat dari nol.
Dipakai oleh: notebook (membuat model .joblib) dan backend Flask (web simulasi).
Studi kasus: Prediksi Tingkat Kemacetan Lalu Lintas di Waktu Tertentu.
"""
import csv
import math
import random

TARGET = "Kemacetan"
CLASSES = ["Lancar", "Macet"]
ATTRS = ["Waktu", "Hari", "Cuaca", "Acara"]
DOMAINS = {
    "Waktu": ["Pagi", "Siang", "Sore", "Malam"],
    "Hari": ["Hari Kerja", "Akhir Pekan"],
    "Cuaca": ["Cerah", "Hujan"],
    "Acara": ["Tidak", "Ada"],
}


# ---------------------------------------------------------------- data
def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["No"] = int(r["No"])
    return rows


# ---------------------------------------------------------- perhitungan
def class_counts(rows):
    return {c: sum(1 for r in rows if r[TARGET] == c) for c in CLASSES}


def entropy(counts):
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    return h if h > 1e-12 else 0.0  # hindari -0.0


def majority(counts):
    # seri -> Macet (kelas terakhir pada CLASSES)
    return max(CLASSES, key=lambda c: (counts[c], CLASSES.index(c)))


def analyze_attribute(rows, attr, h_total):
    n = len(rows)
    present = {r[attr] for r in rows}
    values = []
    for v in DOMAINS[attr]:
        if v not in present:
            continue
        sub = [r for r in rows if r[attr] == v]
        cnt = class_counts(sub)
        share = len(sub) / n
        values.append({
            "value": v,
            "n": len(sub),
            "counts": cnt,
            "entropy": entropy(cnt),
            "split_part": (-share * math.log2(share)) if share < 1 else 0.0,
        })
    gain = h_total - sum((x["n"] / n) * x["entropy"] for x in values)
    if abs(gain) < 1e-12:
        gain = 0.0
    split_info = sum(x["split_part"] for x in values)
    ratio = gain / split_info if split_info > 0 else 0.0
    return {"gain": gain, "split_info": split_info, "gain_ratio": ratio, "values": values}


# ---------------------------------------------------------------- pohon
def build_node(rows, available, rng, m, path, logs):
    counts = class_counts(rows)
    n = len(rows)
    nonzero = [c for c in CLASSES if counts[c] > 0]
    if len(nonzero) == 1:
        return {"type": "leaf", "label": nonzero[0], "n": n, "counts": counts, "reason": "pure"}
    if not available:
        return {"type": "leaf", "label": majority(counts), "n": n, "counts": counts, "reason": "no_attr"}

    h_total = entropy(counts)
    candidates = rng.sample(available, min(m, len(available)))
    analysis = {a: analyze_attribute(rows, a, h_total) for a in candidates}
    best = max(candidates, key=lambda a: analysis[a]["gain_ratio"])
    if len(analysis[best]["values"]) < 2:
        return {"type": "leaf", "label": majority(counts), "n": n, "counts": counts, "reason": "no_split"}

    log_id = len(logs)
    logs.append({
        "id": log_id,
        "path": [list(p) for p in path],
        "n": n,
        "counts": counts,
        "entropy": h_total,
        "candidates": candidates,
        "analysis": analysis,
        "best": best,
    })
    node = {"type": "node", "attr": best, "n": n, "counts": counts,
            "log_id": log_id, "majority": majority(counts), "children": []}
    rest = [a for a in available if a != best]
    for item in analysis[best]["values"]:
        v = item["value"]
        sub = [r for r in rows if r[best] == v]
        child = build_node(sub, rest, rng, m, path + [(best, v)], logs)
        node["children"].append({"value": v, "node": child})
    return node


def build_forest(data, n_trees=3, m=2, seed=7):
    rng = random.Random(seed)
    trees = []
    for t in range(n_trees):
        idx = [rng.randrange(len(data)) for _ in range(len(data))]
        sample = [data[i] for i in idx]
        logs = []
        tree = build_node(sample, list(ATTRS), rng, m, [], logs)
        trees.append({
            "index": t + 1,
            "bootstrap_idx": idx,
            "bootstrap_rows": sample,
            "tree": tree,
            "logs": logs,
        })
    return {"meta": {"n_trees": n_trees, "m": m, "seed": seed, "n_data": len(data),
                     "attrs": ATTRS, "classes": CLASSES, "target": TARGET,
                     "domains": DOMAINS},
            "trees": trees}


# ------------------------------------------------------------- prediksi
def predict_tree(node, sample):
    path = []
    while node["type"] == "node":
        v = sample[node["attr"]]
        nxt = next((c for c in node["children"] if c["value"] == v), None)
        if nxt is None:  # nilai tidak pernah muncul pada node ini -> suara mayoritas node
            return node["majority"], path, node
        path.append({"attr": node["attr"], "value": v})
        node = nxt["node"]
    return node["label"], path, node


def predict_forest(forest, sample):
    votes = []
    for t in forest["trees"]:
        label, path, last = predict_tree(t["tree"], sample)
        fallback = None
        if last["type"] == "node":  # nilai tidak ada pada cabang -> mayoritas node
            fallback = {"attr": last["attr"], "value": sample[last["attr"]], "counts": last["counts"]}
        votes.append({"tree": t["index"], "label": label, "path": path, "fallback": fallback})
    counts = {c: sum(1 for v in votes if v["label"] == c) for c in CLASSES}
    return {"votes": votes, "counts": counts, "final": majority(counts)}


# ------------------------------------------------------------ visualisasi
def draw_tree(tree, path=None, title=None, figsize=None, dpi=160):
    """Gambar pohon keputusan dengan matplotlib (opsional: sorot jalur prediksi)."""
    import copy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Ellipse

    tree = copy.deepcopy(tree)  # jangan ubah pohon asli
    path = path or []
    pos, edges, order = {}, [], []
    counter = [0]

    def layout(node, depth, on_path, prefix):
        nid = len(order)
        order.append(node)
        node["_id"] = nid
        node["_on"] = on_path
        if node["type"] == "leaf":
            pos[nid] = (counter[0], -depth)
            counter[0] += 1
            return nid
        kids = []
        for i, ch in enumerate(node["children"]):
            follow = on_path and i_follow(node, ch, prefix)
            cid = layout(ch["node"], depth + 1, follow, prefix + [ch["value"]])
            kids.append((cid, ch["value"], follow))
        xs = [pos[k[0]][0] for k in kids]
        pos[nid] = (sum(xs) / len(xs), -depth)
        for cid, val, follow in kids:
            edges.append((nid, cid, val, follow))
        return nid

    def i_follow(node, ch, prefix):
        step = len(prefix)
        return step < len(path) and path[step]["attr"] == node["attr"] and path[step]["value"] == ch["value"]

    layout(tree, 0, True, [])
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    w = max(6.0, (max(xs) - min(xs) + 1) * 1.75)
    h = max(3.2, (-min(ys) + 1) * 1.45 + 0.6)
    fig, ax = plt.subplots(figsize=figsize or (w, h))
    for a, b, val, follow in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1 - 0.18, y2 + 0.2], color="#E8603C" if follow else "#9ACD6A",
                lw=2.6 if follow else 1.4, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, val, fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3)
    for nid, node in enumerate(order):
        x, y = pos[nid]
        hl = bool(path) and node["_on"]
        if node["type"] == "node":
            ax.add_patch(Ellipse((x, y), 1.45, 0.52, fc="#6AA84F", ec="#E8603C" if hl else "#4E8A38",
                                 lw=2.4 if hl else 1, zorder=2))
            ax.text(x, y, node["attr"], color="white", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=4)
        else:
            color = "#E05555" if node["label"] == "Macet" else "#92D050"
            ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.2), 0.76, 0.4,
                                        boxstyle="round,pad=0.02,rounding_size=0.08", fc=color,
                                        ec="#E8603C" if hl else color, lw=2.4 if hl else 1, zorder=2))
            ax.text(x, y, node["label"], color="white", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=4)
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, style="italic")
    fig.tight_layout()
    return fig
