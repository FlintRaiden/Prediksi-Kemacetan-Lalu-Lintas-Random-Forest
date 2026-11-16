# Random Forest — Prediksi Tingkat Kemacetan Lalu Lintas

Perhitungan manual + web simulasi Random Forest (Gain Ratio / C4.5) untuk studi kasus
**Prediksi Tingkat Kemacetan Lalu Lintas di Waktu Tertentu**.

## Struktur
```
notebook/Random_Forest_Kemacetan.ipynb   <- jalankan ini untuk MENGHASILKAN model .joblib
models/rf_kemacetan.joblib               <- model (hasil notebook), dimuat backend web
public/model.json                        <- ekspor isi model yang sama (cadangan statis untuk web)
simulation/rf_engine.py                  <- mesin Random Forest dari nol (dipakai notebook & web)
api/index.py                             <- backend Flask (/api/forest, /api/predict)
public/index.html                        <- tampilan web simulasi
data/kemacetan.csv                       <- dataset 14 baris
vercel.json, requirements.txt            <- konfigurasi deploy Vercel
```

## Alur kerja
1. Buka `notebook/Random_Forest_Kemacetan.ipynb` (Colab atau Jupyter lokal), **Run all**.
   Notebook menulis `simulation/rf_engine.py`, membangun hutan (3 pohon, m = 2, seed = 3),
   lalu menyimpan `models/rf_kemacetan.joblib`.
   - Di Colab: unduh `rf_kemacetan.joblib` (sel terakhir) lalu taruh di folder `models/` project.
2. Jalankan web lokal:
   ```bash
   pip install -r requirements.txt
   python api/index.py        # buka http://127.0.0.1:5000
   ```
3. Deploy ke Vercel: push ke GitHub, import repo di Vercel (framework: Other). `public/` disajikan statis,
   `/api/*` ditangani `api/index.py`.

## Deploy & pemecahan masalah (Vercel)
- Commit **semua** folder, termasuk `models/rf_kemacetan.joblib` dan `public/model.json` (sudah aman dengan `.gitignore` yang disediakan).
- Cek `https://<domain-kamu>.vercel.app/api/health`: `"ok": true` berarti backend + joblib termuat. Jika `false`, isi `errors` menunjukkan sebabnya.
- Jika API gagal, halaman tetap menampilkan data dan prediksi dari `model.json` (badge berubah menjadi "model.json (ekspor joblib, statis)"), tetapi pengaturan seed/jumlah pohon/m tidak bisa diubah.
- Setiap kali notebook dijalankan ulang, commit ulang kedua file model (`.joblib` dan `.json`).

## Catatan
- Seed default `3` menghasilkan pohon yang sama dengan perhitungan manual di laporan.
- Mengubah jumlah pohon / m / seed di web membangun ulang hutan lewat `rf_engine` (tanpa melatih ulang notebook).
- Aturan tie: bila suara seri, kelas `Macet` dipilih. Nilai atribut yang tidak ada pada suatu cabang
  akan memakai suara mayoritas node tersebut.
- Dataset kecil (14 baris) hanya untuk ilustrasi perhitungan manual, bukan untuk klaim performa.
