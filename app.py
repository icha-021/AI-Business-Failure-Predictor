import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="AI Business Failure Predictor", page_icon="🏢", layout="wide")

# ================================================================
# KONSTANTA
# ================================================================
DB_PATH = Path("data/history.db")
DB_PATH.parent.mkdir(exist_ok=True)

COEFS = {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0}
META = {
    "X1": ("Likuiditas", 0.15),
    "X2": ("Laba Ditahan", 0.10),
    "X3": ("Profitabilitas", 0.05),
    "X4": ("Solvabilitas", 0.50),
    "X5": ("Efisiensi", 0.50),
}
BENCHMARKS = {
    "Manufaktur":  {"X1": 0.20, "X2": 0.15, "X3": 0.08, "X4": 1.2, "X5": 1.5},
    "Perdagangan": {"X1": 0.15, "X2": 0.12, "X3": 0.07, "X4": 1.0, "X5": 2.0},
    "Jasa":        {"X1": 0.20, "X2": 0.18, "X3": 0.10, "X4": 1.5, "X5": 1.0},
    "Teknologi":   {"X1": 0.25, "X2": 0.10, "X3": 0.12, "X4": 1.8, "X5": 0.8},
    "Konstruksi":  {"X1": 0.12, "X2": 0.08, "X3": 0.06, "X4": 0.8, "X5": 1.2},
    "Pertanian":   {"X1": 0.18, "X2": 0.10, "X3": 0.07, "X4": 0.9, "X5": 1.0},
    "Lainnya":     {"X1": 0.18, "X2": 0.12, "X3": 0.08, "X4": 1.1, "X5": 1.2},
}
REQUIRED_COLS = [
    "nama", "sektor", "tahun", "total_aset", "aset_lancar",
    "liabilitas_jangka_pendek", "total_liabilitas", "laba_ditahan",
    "ebit", "penjualan", "ekuitas",
]

# ================================================================
# DATABASE
# ================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT, sektor TEXT, tahun INTEGER, timestamp TEXT,
            z_score REAL, probabilitas REAL, label TEXT, komponen_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_score(nama, sektor, tahun, z, prob, label, komponen):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO scores
        (nama, sektor, tahun, timestamp, z_score, probabilitas, label, komponen_json)
        VALUES (?,?,?,?,?,?,?,?)""",
        (nama, sektor, tahun, datetime.now().isoformat(),
         z, prob, label, json.dumps(komponen)))
    conn.commit()
    conn.close()


def get_history(nama=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if nama:
        rows = conn.execute("SELECT * FROM scores WHERE nama=? ORDER BY timestamp",
                            (nama,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM scores ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_companies():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT nama, MAX(timestamp) AS last FROM scores GROUP BY nama"
    ).fetchall()
    conn.close()
    return [{"nama": r[0], "last": r[1]} for r in rows]


# ================================================================
# SAMPEL DATA (MODE UJI COBA)
# ================================================================
@st.cache_data
def get_sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"nama": "PT Maju Jaya", "sektor": "Manufaktur", "tahun": 2023,
         "total_aset": 500e9, "aset_lancar": 200e9, "liabilitas_jangka_pendek": 80e9,
         "total_liabilitas": 120e9, "laba_ditahan": 100e9, "ebit": 60e9,
         "penjualan": 700e9, "ekuitas": 380e9},
        {"nama": "PT Teknologi Nusantara", "sektor": "Teknologi", "tahun": 2023,
         "total_aset": 400e9, "aset_lancar": 220e9, "liabilitas_jangka_pendek": 50e9,
         "total_liabilitas": 50e9, "laba_ditahan": 150e9, "ebit": 90e9,
         "penjualan": 450e9, "ekuitas": 350e9},
        {"nama": "PT Sejahtera Abadi", "sektor": "Perdagangan", "tahun": 2023,
         "total_aset": 200e9, "aset_lancar": 80e9, "liabilitas_jangka_pendek": 60e9,
         "total_liabilitas": 120e9, "laba_ditahan": 25e9, "ebit": 12e9,
         "penjualan": 250e9, "ekuitas": 80e9},
        {"nama": "PT Rentan Modal", "sektor": "Jasa", "tahun": 2023,
         "total_aset": 150e9, "aset_lancar": 60e9, "liabilitas_jangka_pendek": 30e9,
         "total_liabilitas": 70e9, "laba_ditahan": 20e9, "ebit": 12e9,
         "penjualan": 200e9, "ekuitas": 80e9},
        {"nama": "PT Krisis Finance", "sektor": "Konstruksi", "tahun": 2023,
         "total_aset": 300e9, "aset_lancar": 50e9, "liabilitas_jangka_pendek": 120e9,
         "total_liabilitas": 280e9, "laba_ditahan": -20e9, "ebit": -15e9,
         "penjualan": 150e9, "ekuitas": 20e9},
    ])


@st.cache_data
def generate_sample_excel() -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        get_sample_df().to_excel(w, sheet_name="Data Keuangan", index=False)
    return buf.getvalue()


# ================================================================
# MODEL & LOGIKA
# ================================================================
def compute_ratios(r: dict) -> dict:
    ta = float(r["total_aset"]) or 1.0
    tl = float(r["total_liabilitas"]) or 1.0
    return {
        "X1": (float(r["aset_lancar"]) - float(r["liabilitas_jangka_pendek"])) / ta,
        "X2": float(r["laba_ditahan"]) / ta,
        "X3": float(r["ebit"]) / ta,
        "X4": float(r["ekuitas"]) / tl,
        "X5": float(r["penjualan"]) / ta,
    }


def predict(r: dict) -> dict:
    komp = compute_ratios(r)
    z = sum(COEFS[k] * komp[k] for k in COEFS)
    prob = float(1 / (1 + np.exp(1.5 * (z - 2.4))))
    label = "Aman" if prob < 0.30 else "Waspada" if prob < 0.60 else "Bahaya"
    return {"z_score": z, "probabilitas": prob, "label": label, "komponen": komp}


def explain(komp: dict) -> pd.DataFrame:
    rows = []
    for k, (nama, baseline) in META.items():
        val = komp[k]
        rows.append({
            "Kode": k,
            "Faktor": nama,
            "Nilai": round(val, 4),
            "Kontribusi": round(COEFS[k] * val, 4),
            "Status": "Positif" if val >= baseline else "Negatif",
        })
    return pd.DataFrame(rows).sort_values("Kontribusi", key=abs, ascending=False)


def recommend(komp: dict) -> list:
    recs = []
    if komp["X1"] < 0.10:
        recs.append(("Tinggi", "Likuiditas",
                     "Percepat penagihan piutang, negosiasi ulang jatuh tempo utang "
                     "jangka pendek, atau restrukturisasi kewajiban lancar."))
    if komp["X2"] < 0.05:
        recs.append(("Sedang", "Laba Ditahan",
                     "Perkuat laba ditahan melalui efisiensi operasional dan "
                     "kebijakan dividen yang konservatif."))
    if komp["X3"] < 0:
        recs.append(("Tinggi", "Profitabilitas",
                     "Analisis biaya per lini produk; identifikasi unit yang tidak "
                     "menguntungkan untuk dioptimalkan atau dihentikan."))
    if komp["X4"] < 0.5:
        recs.append(("Tinggi", "Struktur Modal",
                     "Kurangi ketergantungan utang berbunga; pertimbangkan injeksi "
                     "ekuitas atau restrukturisasi utang."))
    if komp["X5"] < 0.5:
        recs.append(("Sedang", "Efisiensi Aset",
                     "Optimalkan utilisasi aset; dorong volume penjualan atau "
                     "divestasi aset tidak produktif."))
    if not recs:
        recs.append(("Rendah", "Pemantauan",
                     "Rasio dalam kondisi baik. Lanjutkan pemantauan kuartalan."))
    return recs


def compare_industry(komp: dict, sektor: str) -> pd.DataFrame:
    bench = BENCHMARKS.get(sektor, BENCHMARKS["Lainnya"])
    return pd.DataFrame([{
        "Rasio": k,
        "Perusahaan": round(komp[k], 3),
        "Rata-rata Industri": bench[k],
        "Selisih": round(komp[k] - bench[k], 3),
        "Status": "▲ Di atas" if komp[k] >= bench[k] else "▼ Di bawah",
    } for k in ["X1", "X2", "X3", "X4", "X5"]])


# ================================================================
# GENERATOR PDF
# ================================================================
def generate_pdf(inp: dict, res: dict, df_f: pd.DataFrame, recs: list) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    s = getSampleStyleSheet()
    story = [Paragraph("Laporan Analisis Risiko Kebangkrutan", s["Title"]), Spacer(1, 12)]

    info = [["Perusahaan", inp["nama"]],
            ["Sektor", inp["sektor"]],
            ["Tahun Data", str(inp["tahun"])],
            ["Tanggal", datetime.now().strftime("%d %B %Y %H:%M")]]
    t = Table(info, colWidths=[120, 400])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [t, Spacer(1, 14)]

    story.append(Paragraph("<b>Skor Risiko</b>", s["Heading2"]))
    cmap = {"Aman": colors.green, "Waspada": colors.orange, "Bahaya": colors.red}
    score = [["Metrik", "Nilai"],
             ["Skor Risiko", f"{res['probabilitas']*100:.1f}%"],
             ["Status", res["label"]],
             ["Altman Z-Score", f"{res['z_score']:.2f}"]]
    ts = Table(score, colWidths=[150, 150])
    ts.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (1, 2), (1, 2), cmap[res["label"]]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story += [ts, Spacer(1, 14)]

    story.append(Paragraph("<b>Faktor Penyebab</b>", s["Heading2"]))
    fdata = [["Faktor", "Nilai", "Kontribusi", "Status"]] + \
            df_f[["Faktor", "Nilai", "Kontribusi", "Status"]].astype(str).values.tolist()
    tf = Table(fdata)
    tf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [tf, Spacer(1, 14)]

    story.append(Paragraph("<b>Saran Perbaikan</b>", s["Heading2"]))
    for i, (prio, area, saran) in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. <b>[{prio}] {area}</b>", s["Normal"]))
        story.append(Paragraph(saran, s["Normal"]))
        story.append(Spacer(1, 6))

    story += [Spacer(1, 20),
              Paragraph("<i>Disclaimer: Laporan ini bukan nasihat keuangan resmi.</i>",
                        s["Italic"])]
    doc.build(story)
    return buf.getvalue()


# ================================================================
# RENDER HASIL
# ================================================================
def render_result(inp: dict, res: dict):
    icon = {"Aman": "🟢", "Waspada": "🟡", "Bahaya": "🔴"}[res["label"]]
    st.markdown(f"### Hasil — **{inp['nama']}** ({inp['sektor']}, {inp['tahun']})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor Risiko", f"{res['probabilitas']*100:.1f}%")
    c2.metric("Status", f"{icon} {res['label']}")
    c3.metric("Altman Z-Score", f"{res['z_score']:.2f}")
    st.divider()

    df_f = explain(res["komponen"])
    st.subheader("🔍 Faktor Penyebab")
    fig = px.bar(df_f, x="Kontribusi", y="Faktor", orientation="h", color="Status",
                 color_discrete_map={"Positif": "#10B981", "Negatif": "#EF4444"})
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_f, use_container_width=True, hide_index=True)

    st.subheader("🏭 Perbandingan Industri")
    st.dataframe(compare_industry(res["komponen"], inp["sektor"]),
                 use_container_width=True, hide_index=True)

    st.subheader("💡 Saran Perbaikan")
    recs = recommend(res["komponen"])
    for prio, area, saran in recs:
        st.markdown(f"**{prio} — {area}**")
        st.info(saran)

    st.subheader("📄 Laporan PDF")
    st.download_button(
        "⬇️ Unduh Laporan PDF",
        data=generate_pdf(inp, res, df_f, recs),
        file_name=f"laporan_{inp['nama'].replace(' ', '_')}_{inp['tahun']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    save_score(inp["nama"], inp["sektor"], inp["tahun"],
               res["z_score"], res["probabilitas"], res["label"], res["komponen"])


# ================================================================
# HALAMAN
# ================================================================
def page_beranda():
    st.title("🏢 AI Business Failure Predictor")
    st.markdown("Prediksi risiko kebangkrutan perusahaan berbasis AI. "
                "Input data keuangan → skor risiko, penjelasan faktor, saran perbaikan, "
                "dan laporan PDF siap dibagikan ke bank/investor.")

    st.divider()
    st.subheader("🚀 Mode Uji Coba")
    st.info("Belum punya data? Unduh sampel berisi **5 perusahaan** "
            "(2 Aman, 2 Waspada, 1 Bahaya) untuk mencoba semua fitur.")
    st.download_button(
        "📥 Unduh Sampel Data (Excel)",
        data=generate_sample_excel(),
        file_name="sample_financials.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown("""
    **Cara pakai:** 1) Unduh sampel → 2) Buka **📊 Prediksi** → 3) Upload file →
    4) Pilih perusahaan → 5) Hitung skor → 6) Unduh PDF.
    """)

    st.divider()
    st.subheader("Fitur Utama")
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor Risiko", "0–100%", "Aman/Waspada/Bahaya")
    c2.metric("Explainability", "Faktor", "Naik / turun")
    c3.metric("Laporan PDF", "Siap audit", "Bank & investor")

    st.caption("⚠️ Alat bantu analisis, bukan nasihat keuangan resmi.")


def page_prediksi():
    st.title("📊 Prediksi Risiko Kebangkrutan")
    tab_excel, tab_form = st.tabs(["📁 Upload Excel", "✍️ Form Manual"])

    with tab_excel:
        up = st.file_uploader("Upload file .xlsx", type=["xlsx"])
        if up:
            try:
                df = pd.read_excel(up)
            except Exception as e:
                st.error(f"Gagal membaca: {e}"); return
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error(f"Kolom hilang: {missing}"); return
            st.success(f"Dimuat: {len(df)} perusahaan.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            idx = st.selectbox("Pilih perusahaan:", df.index,
                               format_func=lambda i: df.loc[i, "nama"])
            if st.button("🎯 Hitung Skor", key="btn_x"):
                inp = df.loc[idx].to_dict()
                render_result(inp, predict(inp))

    with tab_form:
        sample = get_sample_df().iloc[0].to_dict()
        with st.form("form_manual"):
            nama = st.text_input("Nama", sample["nama"])
            sektor = st.selectbox("Sektor", list(BENCHMARKS.keys()))
            tahun = st.number_input("Tahun", 2000, 2030, 2023)
            c1, c2 = st.columns(2)
            with c1:
                total_aset = st.number_input("Total Aset", value=float(sample["total_aset"]))
                aset_lancar = st.number_input("Aset Lancar", value=float(sample["aset_lancar"]))
                liab_jp = st.number_input("Liabilitas Jangka Pendek",
                                          value=float(sample["liabilitas_jangka_pendek"]))
                laba_ditahan = st.number_input("Laba Ditahan",
                                               value=float(sample["laba_ditahan"]))
            with c2:
                total_liab = st.number_input("Total Liabilitas",
                                             value=float(sample["total_liabilitas"]))
                ebit = st.number_input("EBIT", value=float(sample["ebit"]))
                penjualan = st.number_input("Penjualan", value=float(sample["penjualan"]))
                ekuitas = st.number_input("Ekuitas", value=float(sample["ekuitas"]))
            submit = st.form_submit_button("🎯 Hitung Skor", use_container_width=True)

        if submit:
            if total_aset <= 0:
                st.error("Total Aset harus > 0"); return
            inp = {"nama": nama, "sektor": sektor, "tahun": tahun,
                   "total_aset": total_aset, "aset_lancar": aset_lancar,
                   "liabilitas_jangka_pendek": liab_jp,
                   "total_liabilitas": total_liab, "laba_ditahan": laba_ditahan,
                   "ebit": ebit, "penjualan": penjualan, "ekuitas": ekuitas}
            render_result(inp, predict(inp))


def page_riwayat():
    st.title("📈 Riwayat Skor")
    companies = list_companies()
    if not companies:
        st.info("Belum ada riwayat. Hitung skor di halaman **Prediksi** dulu.")
        return
    nama = st.selectbox("Perusahaan:", [c["nama"] for c in companies])
    df = pd.DataFrame(get_history(nama))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["skor_%"] = df["probabilitas"] * 100

    fig = px.line(df, x="timestamp", y="skor_%", markers=True,
                  title=f"Tren Skor Risiko — {nama}")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Aman")
    fig.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Bahaya")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["timestamp", "tahun", "sektor", "skor_%", "z_score", "label"]],
                 use_container_width=True, hide_index=True)

    if len(df) >= 2:
        d = df["skor_%"].iloc[-1] - df["skor_%"].iloc[-2]
        if d > 10:
            st.error(f"⚠️ **Notifikasi:** skor naik {d:.1f} poin dari pengukuran sebelumnya!")
        elif d > 0:
            st.warning(f"Skor naik {d:.1f} poin.")
        elif d < -10:
            st.success(f"✅ Skor turun {abs(d):.1f} poin — membaik.")


def page_simulasi():
    st.title("🔄 Simulasi What-If")
    df = get_sample_df()
    idx = st.selectbox("Perusahaan:", df.index, format_func=lambda i: df.loc[i, "nama"])
    base = df.loc[idx].to_dict()
    base_res = predict(base)

    c1, c2 = st.columns(2)
    with c1:
        d_penjualan = st.slider("Perubahan Penjualan (%)", -50, 50, 0)
        d_ebit = st.slider("Perubahan EBIT (%)", -50, 100, 0)
        d_liab = st.slider("Perubahan Total Liabilitas (%)", -50, 100, 0)
    with c2:
        d_aset = st.slider("Perubahan Aset Lancar (%)", -50, 100, 0)
        d_liab_jp = st.slider("Perubahan Liabilitas Jangka Pendek (%)", -50, 100, 0)

    sim = dict(base)
    sim["aset_lancar"] = base["aset_lancar"] * (1 + d_aset / 100)
    sim["liabilitas_jangka_pendek"] = base["liabilitas_jangka_pendek"] * (1 + d_liab_jp / 100)
    sim["total_liabilitas"] = base["total_liabilitas"] * (1 + d_liab / 100)
    sim["ebit"] = base["ebit"] * (1 + d_ebit / 100)
    sim["penjualan"] = base["penjualan"] * (1 + d_penjualan / 100)
    sim["ekuitas"] = base["total_aset"] - sim["total_liabilitas"]
    sim_res = predict(sim)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline", f"{base_res['probabilitas']*100:.1f}%", base_res["label"])
    delta = (sim_res["probabilitas"] - base_res["probabilitas"]) * 100
    c2.metric("Simulasi", f"{sim_res['probabilitas']*100:.1f}%",
              f"{delta:+.1f} pp", delta_color="inverse")
    c3.metric("Status Baru", sim_res["label"])

    b = explain(base_res["komponen"]).set_index("Faktor")["Kontribusi"]
    s = explain(sim_res["komponen"]).set_index("Faktor")["Kontribusi"]
    cmp_df = pd.DataFrame({"Baseline": b, "Simulasi": s}).reset_index()
    fig = px.bar(cmp_df, x="Faktor", y=["Baseline", "Simulasi"], barmode="group",
                 title="Kontribusi Faktor: Baseline vs Simulasi")
    st.plotly_chart(fig, use_container_width=True)


def page_multi():
    st.title("🏢 Pemantauan Multi Perusahaan")
    up = st.file_uploader("Upload Excel portofolio (.xlsx)", type=["xlsx"],
                          key="multi_up")
    if not up:
        st.info("Upload file Excel berisi banyak perusahaan.")
        return
    try:
        df = pd.read_excel(up)
    except Exception as e:
        st.error(f"Gagal membaca: {e}"); return
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Kolom hilang: {missing}"); return

    rows, errors = [], []
    for i, r in df.iterrows():
        try:
            res = predict(r.to_dict())
            rows.append({"Perusahaan": r["nama"], "Sektor": r["sektor"],
                         "Tahun": r["tahun"],
                         "Skor (%)": round(res["probabilitas"] * 100, 1),
                         "Status": res["label"],
                         "Z-Score": round(res["z_score"], 2)})
        except Exception as e:
            errors.append(f"Baris {i+2}: {e}")

    if errors:
        with st.expander(f"⚠️ {len(errors)} baris gagal"):
            for e in errors: st.text(e)

    if not rows: return
    hasil = pd.DataFrame(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(hasil))
    c2.metric("🟢 Aman", int((hasil["Status"] == "Aman").sum()))
    c3.metric("🟡 Waspada", int((hasil["Status"] == "Waspada").sum()))
    c4.metric("🔴 Bahaya", int((hasil["Status"] == "Bahaya").sum()))

    bahaya = hasil[hasil["Status"] == "Bahaya"]
    if not bahaya.empty:
        st.error(f"⚠️ **Notifikasi:** {len(bahaya)} perusahaan berstatus BAHAYA — "
                 f"{', '.join(bahaya['Perusahaan'].tolist())}")

    st.dataframe(hasil.sort_values("Skor (%)", ascending=False),
                 use_container_width=True, hide_index=True)

    fig = px.bar(hasil.sort_values("Skor (%)"), x="Skor (%)", y="Perusahaan",
                 color="Status", orientation="h",
                 color_discrete_map={"Aman": "#10B981", "Waspada": "#F59E0B",
                                     "Bahaya": "#EF4444"},
                 title="Peringkat Risiko Portofolio")
    fig.update_layout(height=max(300, 40 * len(hasil)))
    st.plotly_chart(fig, use_container_width=True)


# ================================================================
# NAVIGASI UTAMA
# ================================================================
init_db()

with st.sidebar:
    st.markdown("### 🏢 Navigasi")
    page = st.radio("Menu", [
        "🏠 Beranda", "📊 Prediksi", "📈 Riwayat",
        "🔄 Simulasi", "🏢 Multi Perusahaan",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Mode Uji Coba**")
    st.download_button(
        "📥 Unduh Sampel Data",
        data=generate_sample_excel(),
        file_name="sample_financials.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("Altman Z-Score · baseline model")

{"🏠 Beranda": page_beranda,
 "📊 Prediksi": page_prediksi,
 "📈 Riwayat": page_riwayat,
 "🔄 Simulasi": page_simulasi,
 "🏢 Multi Perusahaan": page_multi}[page]()
