# 🏢 AI Business Failure Predictor

Prediksi risiko kebangkrutan perusahaan dengan **Python + Streamlit**. Satu file, semua fitur.

## ✨ Fitur
- 📁 Input via **Excel upload** atau **form manual**
- 🎯 **Skor risiko** 0–100% + label **Aman / Waspada / Bahaya** (Altman Z-Score)
- 🔍 **Explainability** — kontribusi tiap rasio keuangan
- 💡 **Saran perbaikan** konkret per area
- 📄 **Laporan PDF** siap dibagikan ke bank/investor
- 📈 **Riwayat skor** dalam grafik + notifikasi otomatis
- 🏭 **Benchmark industri** (7 sektor)
- 🔄 **Simulasi What-If** interaktif
- 🏢 **Multi-perusahaan dashboard** untuk bank/investor

## 🚀 Mode Uji Coba
Tombol **📥 Unduh Sampel Data** ada di **sidebar** dan **halaman Beranda**.
File berisi **5 perusahaan** dengan profil risiko berbeda (2 Aman, 2 Waspada, 1 Bahaya).

## 🛠 Instalasi
```bash
git clone https://github.com/<username>/ai-business-failure-predictor.git
cd ai-business-failure-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
