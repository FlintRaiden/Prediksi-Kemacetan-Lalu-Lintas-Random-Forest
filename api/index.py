"""Backend Flask - Web Simulasi Random Forest (Prediksi Kemacetan Lalu Lintas).
Model dimuat dari models/rf_kemacetan.joblib (dihasilkan oleh notebook).
Jika pengguna mengubah seed / jumlah pohon / m, hutan dibangun ulang oleh rf_engine.
Pemuatan dibuat malas (lazy) + berlapis agar error import/file tidak mematikan seluruh API.
"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT_CANDIDATES = [HERE.parent, Path.cwd(), Path("/var/task")]

app = Flask(__name__)
_state = {"rf": None, "model": None, "root": None, "errors": []}


def _find(rel):
    for root in ROOT_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    return None


def _load():
    """Muat engine + model sekali saja. Catat error, jangan crash."""
    if _state["rf"] is not None:
        return
    try:
        eng = _find("simulation/rf_engine.py")
        if eng is None:
            raise FileNotFoundError("simulation/rf_engine.py tidak ditemukan")
        spec = importlib.util.spec_from_file_location("rf_engine", eng)
        rf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf)
        _state["rf"], _state["root"] = rf, eng.parent.parent
    except Exception:
        _state["errors"].append("engine: " + traceback.format_exc(limit=3))
        return

    rf = _state["rf"]
    try:
        import joblib
        mp = _find("models/rf_kemacetan.joblib")
        if mp is None:
            raise FileNotFoundError("models/rf_kemacetan.joblib tidak ditemukan")
        _state["model"] = joblib.load(mp)
        _state["model_src"] = "joblib"
    except Exception:
        _state["errors"].append("joblib: " + traceback.format_exc(limit=3))
        try:  # cadangan: bangun dari CSV dengan parameter default
            data = rf.load_csv(_find("data/kemacetan.csv"))
            forest = rf.build_forest(data, n_trees=3, m=2, seed=3)
            _state["model"] = {"forest": forest, "dataset": data,
                               "meta": {"n_trees": 3, "m": 2, "seed": 3}}
            _state["model_src"] = "csv"
        except Exception:
            _state["errors"].append("csv: " + traceback.format_exc(limit=3))


def _int(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _get_forest(seed, n_trees, m):
    rf, model = _state["rf"], _state["model"]
    meta = model["meta"]
    if (seed, n_trees, m) == (meta["seed"], meta["n_trees"], meta["m"]):
        return model["forest"], _state.get("model_src", "joblib")
    return rf.build_forest(model["dataset"], n_trees=n_trees, m=m, seed=seed), "engine"


def _ready():
    _load()
    if _state["rf"] is None or _state["model"] is None:
        return jsonify({"error": "Backend belum siap", "detail": _state["errors"]}), 500
    return None


@app.get("/api/health")
def health():
    _load()
    return jsonify({"ok": _state["rf"] is not None and _state["model"] is not None,
                    "model_src": _state.get("model_src"), "errors": _state["errors"],
                    "cwd": os.getcwd(), "here": str(HERE),
                    "files": sorted(p.name for p in HERE.parent.iterdir())})


@app.get("/api/forest")
def api_forest():
    bad = _ready()
    if bad:
        return bad
    rf, meta = _state["rf"], _state["model"]["meta"]
    seed = _int("seed", meta["seed"], 0, 999999)
    n_trees = _int("n_trees", meta["n_trees"], 1, 15)
    m = _int("m", meta["m"], 1, len(rf.ATTRS))
    forest, source = _get_forest(seed, n_trees, m)
    return jsonify({"source": source, "meta": forest["meta"],
                    "dataset": _state["model"]["dataset"], "trees": forest["trees"]})


@app.post("/api/predict")
def api_predict():
    bad = _ready()
    if bad:
        return bad
    rf, meta = _state["rf"], _state["model"]["meta"]
    body = request.get_json(silent=True) or {}
    sample = body.get("sample", {})
    for a in rf.ATTRS:
        if sample.get(a) not in rf.DOMAINS[a]:
            return jsonify({"error": f"Nilai atribut '{a}' tidak valid"}), 400
    seed = int(body.get("seed", meta["seed"]))
    n_trees = max(1, min(15, int(body.get("n_trees", meta["n_trees"]))))
    m = max(1, min(len(rf.ATTRS), int(body.get("m", meta["m"]))))
    forest, source = _get_forest(seed, n_trees, m)
    return jsonify({"source": source, **rf.predict_forest(forest, sample)})


@app.get("/")
def index():  # hanya dipakai saat lokal; di Vercel public/ disajikan statis
    return send_from_directory(HERE.parent / "public", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
