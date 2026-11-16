"""Backend Flask - Web Simulasi Random Forest (Prediksi Kemacetan Lalu Lintas).
Model dimuat dari models/rf_kemacetan.joblib (dihasilkan oleh notebook).
Jika pengguna mengubah seed / jumlah pohon / m, hutan dibangun ulang oleh rf_engine.
"""
import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simulation"))
import rf_engine as rf  # noqa: E402

app = Flask(__name__)
MODEL = joblib.load(ROOT / "models" / "rf_kemacetan.joblib")
DATA = MODEL["dataset"]
META = MODEL["meta"]


def _int(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def get_forest(seed, n_trees, m):
    if (seed, n_trees, m) == (META["seed"], META["n_trees"], META["m"]):
        return MODEL["forest"], "joblib"
    return rf.build_forest(DATA, n_trees=n_trees, m=m, seed=seed), "engine"


@app.get("/api/forest")
def api_forest():
    seed = _int("seed", META["seed"], 0, 999999)
    n_trees = _int("n_trees", META["n_trees"], 1, 15)
    m = _int("m", META["m"], 1, len(rf.ATTRS))
    forest, source = get_forest(seed, n_trees, m)
    return jsonify({"source": source, "meta": forest["meta"], "dataset": DATA, "trees": forest["trees"]})


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    sample = body.get("sample", {})
    for a in rf.ATTRS:
        if sample.get(a) not in rf.DOMAINS[a]:
            return jsonify({"error": f"Nilai atribut '{a}' tidak valid"}), 400
    seed = int(body.get("seed", META["seed"]))
    n_trees = max(1, min(15, int(body.get("n_trees", META["n_trees"]))))
    m = max(1, min(len(rf.ATTRS), int(body.get("m", META["m"]))))
    forest, source = get_forest(seed, n_trees, m)
    return jsonify({"source": source, **rf.predict_forest(forest, sample)})


@app.get("/")
def index():  # hanya dipakai saat dijalankan lokal; di Vercel public/ disajikan statis
    return send_from_directory(ROOT / "public", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
