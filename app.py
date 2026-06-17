"""
app.py — Aplikasi Prediksi Harga Mobil Bekas
Model: Stacking Regressor (CatBoost + LightGBM + XGBoost -> Ridge, Optuna-tuned)

Cara menjalankan:
    streamlit run app.py

Berkas pendukung (harus berada di folder yang sama):
    - model_bundle.pkl   (dihasilkan dari notebook)
    - fe_utils.py
"""
import joblib
import numpy as np
import streamlit as st
from fe_utils import transform_one, extract_engine_info

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Gaya antarmuka (CSS)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #f4f6f9;
    --surface:   #ffffff;
    --ink:       #0f172a;
    --ink-soft:  #475569;
    --muted:     #94a3b8;
    --line:      #e2e8f0;
    --accent:    #1d4ed8;
    --accent-dk: #1e3a8a;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); }

/* sembunyikan elemen bawaan Streamlit */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* batasi lebar konten */
.block-container {
    max-width: 780px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* ---------- Header ---------- */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 32px 34px;
    color: #fff;
    margin-bottom: 28px;
    box-shadow: 0 14px 34px -16px rgba(15, 23, 42, 0.5);
}
.app-header h1 {
    font-size: 1.62rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    color: #fff;
}
.app-header p {
    margin: 9px 0 0;
    font-size: 0.92rem;
    color: #cbd5e1;
    font-weight: 400;
}
.app-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 6px 13px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    color: #e2e8f0;
}

/* ---------- Judul seksi ---------- */
.section-title {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--ink-soft);
    margin: 8px 0 2px;
}

/* ---------- Input ---------- */
.stTextInput label, .stNumberInput label, .stSelectbox label {
    font-weight: 500 !important;
    color: var(--ink) !important;
    font-size: 0.86rem !important;
}
.stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
}

/* ---------- Tombol ---------- */
.stButton > button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.72rem 1rem;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
    box-shadow: 0 8px 18px -9px rgba(29, 78, 216, 0.65);
}
.stButton > button:hover {
    background: var(--accent-dk);
    transform: translateY(-1px);
}

/* ---------- Chip spesifikasi mesin ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 2px; }
.chip {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 11px 15px;
    min-width: 104px;
}
.chip .k {
    font-size: 0.68rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.chip .v { font-size: 1.02rem; font-weight: 600; color: var(--ink); margin-top: 2px; }

/* ---------- Kartu hasil ---------- */
.result-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 8px;
    box-shadow: 0 14px 34px -22px rgba(15, 23, 42, 0.3);
}
.result-vehicle {
    font-size: 0.9rem; font-weight: 500; color: var(--ink-soft); margin-bottom: 14px;
}
.result-label {
    font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--muted); font-weight: 600;
}
.result-price {
    font-size: 2.5rem; font-weight: 700; color: var(--ink);
    letter-spacing: -0.025em; margin: 4px 0 2px;
}
.result-idr { font-size: 0.92rem; color: var(--ink-soft); }

.note { font-size: 0.8rem; color: var(--muted); line-height: 1.6; margin-top: 26px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>Prediksi Harga Mobil Bekas</h1>
    <p>Estimasi harga kendaraan berbasis machine learning dari spesifikasi teknis.</p>
    <span class="app-badge">STACKING REGRESSOR &nbsp;·&nbsp; CATBOOST &nbsp;·&nbsp; LIGHTGBM &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; RIDGE</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Muat model
# ----------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("model_bundle.pkl")
    return bundle["model"], bundle["artifacts"]

try:
    model, A = load_bundle()
except FileNotFoundError:
    st.error("Berkas **model_bundle.pkl** tidak ditemukan. "
             "Jalankan terlebih dahulu cell penyimpanan model pada notebook.")
    st.stop()


# ----------------------------------------------------------------------
# Utilitas tampilan
# ----------------------------------------------------------------------
def fmt(v, suffix=""):
    """Format nilai mesin secara rapi; tampilkan tanda strip bila kosong."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        if v.is_integer():
            v = int(v)
    return f"{v}{suffix}"


def grp(n):
    """Pengelompokan ribuan gaya Indonesia (titik sebagai pemisah)."""
    return f"{n:,.0f}".replace(",", ".")


# ----------------------------------------------------------------------
# Input — Spesifikasi Kendaraan
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">Spesifikasi Kendaraan</div>', unsafe_allow_html=True)

brands = A.get("brands", [])
col1, col2 = st.columns(2)
with col1:
    if brands:
        brand = st.selectbox("Merek", brands,
                             index=brands.index("Toyota") if "Toyota" in brands else 0)
    else:
        brand = st.text_input("Merek", "Toyota")
with col2:
    models = A.get("models_by_brand", {}).get(brand, [])
    if models:
        model_name = st.selectbox("Model", models)
    else:
        model_name = st.text_input("Model", "")

col3, col4 = st.columns(2)
with col3:
    model_year = st.number_input("Tahun Produksi", 1980, A["current_year"], 2018, step=1)
with col4:
    mileage = st.number_input("Jarak Tempuh (mil)", 0, 500_000, 50_000, step=1_000)

engine = st.text_input(
    "Spesifikasi Mesin",
    "300.0HP 3.5L V6 Cylinder Engine Gasoline Fuel",
    help="Contoh: '300.0HP 3.5L V6 ...'. Horsepower, displacement (L), dan "
         "konfigurasi diekstrak otomatis. Nilai yang tidak terbaca akan diisi median.",
)

# ----------------------------------------------------------------------
# Hasil ekstraksi mesin
# ----------------------------------------------------------------------
hp, disp, cyl, turbo, config = extract_engine_info(engine)
hp_s, disp_s = fmt(hp), fmt(disp, " L")
cyl_s, config_s = fmt(cyl), fmt(config)
turbo_s = "Ya" if turbo else "Tidak"

st.markdown('<div class="section-title" style="margin-top:18px;">Hasil Ekstraksi Mesin</div>',
            unsafe_allow_html=True)
st.markdown(f"""
<div class="chip-row">
    <div class="chip"><div class="k">Horsepower</div><div class="v">{hp_s}</div></div>
    <div class="chip"><div class="k">Displacement</div><div class="v">{disp_s}</div></div>
    <div class="chip"><div class="k">Silinder</div><div class="v">{cyl_s}</div></div>
    <div class="chip"><div class="k">Turbo</div><div class="v">{turbo_s}</div></div>
    <div class="chip"><div class="k">Konfigurasi</div><div class="v">{config_s}</div></div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Prediksi
# ----------------------------------------------------------------------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

if st.button("Hitung Estimasi Harga", type="primary", use_container_width=True):
    if not str(model_name).strip():
        st.warning("Silakan isi nama model terlebih dahulu.")
        st.stop()

    raw = {"brand": brand, "model": model_name, "model_year": model_year,
           "milage": mileage, "engine": engine}
    X = transform_one(raw, A)
    pred_log = model.predict(X)[0]          # target = log1p(price)
    price = float(np.expm1(pred_log))       # kembalikan ke USD
    price_idr = price * 16_000

    vehicle = f"{brand} {model_name} · {model_year} · {grp(mileage)} mil"
    st.markdown(f"""
    <div class="result-card">
        <div class="result-vehicle">{vehicle}</div>
        <div class="result-label">Estimasi Harga</div>
        <div class="result-price">$ {price:,.0f}</div>
        <div class="result-idr">setara Rp {grp(price_idr)} &nbsp;·&nbsp; kurs asumsi Rp 16.000 / USD</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Lihat 28 fitur yang dihitung"):
        st.dataframe(X.T.rename(columns={0: "nilai"}), use_container_width=True)

# ----------------------------------------------------------------------
# Catatan
# ----------------------------------------------------------------------
st.markdown("""
<p class="note">
Estimasi dihasilkan dari model yang dilatih pada dataset historis dan bersifat indikatif.
Pastikan versi pustaka scikit-learn, catboost, lightgbm, dan xgboost sama dengan
saat pelatihan model agar hasil konsisten.
</p>
""", unsafe_allow_html=True)
