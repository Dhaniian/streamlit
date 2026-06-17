"""
app.py — Streamlit demo Prediksi Harga Mobil Bekas
Model: Stacking Regressor (CatBoost + LightGBM + XGBoost -> Ridge)

Cara jalan:
    streamlit run app.py

Butuh file di folder yang sama:
    - model_bundle.pkl   (dibuat dari notebook, lihat instruksi)
    - fe_utils.py
"""

import joblib
import numpy as np
import streamlit as st

from fe_utils import transform_one, extract_engine_info

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗",
                   layout="centered")


@st.cache_resource
def load_bundle():
    bundle = joblib.load("model_bundle.pkl")
    return bundle["model"], bundle["artifacts"]


try:
    model, A = load_bundle()
except FileNotFoundError:
    st.error("File **model_bundle.pkl** tidak ditemukan. "
             "Jalankan dulu cell penyimpanan di notebook.")
    st.stop()

st.title("🚗 Prediksi Harga Mobil Bekas")
st.caption("Stacking Regressor — CatBoost · LightGBM · XGBoost · Ridge (Optuna-tuned)")

# ---------------- Input ----------------
brands = A.get("brands", [])
col1, col2 = st.columns(2)

with col1:
    if brands:
        brand = st.selectbox("Merek (Brand)", brands,
                             index=brands.index("Toyota") if "Toyota" in brands else 0)
    else:
        brand = st.text_input("Merek (Brand)", "Toyota")

with col2:
    models = A.get("models_by_brand", {}).get(brand, [])
    if models:
        model_name = st.selectbox("Model", models)
    else:
        model_name = st.text_input("Model", "")

col3, col4 = st.columns(2)
with col3:
    model_year = st.number_input("Tahun (Model Year)", 1980, A["current_year"],
                                 2018, step=1)
with col4:
    mileage = st.number_input("Kilometer / Mileage (mi)", 0, 500_000, 50_000,
                              step=1_000)

engine = st.text_input(
    "Spesifikasi Mesin (engine)",
    "300.0HP 3.5L V6 Cylinder Engine Gasoline Fuel",
    help="Contoh: '300.0HP 3.5L V6 ...'. HP, displacement (L), dan konfigurasi "
         "akan diekstrak otomatis. Jika tidak terbaca, sistem memakai median.")

# Tampilkan hasil parsing mesin
hp, disp, cyl, turbo, config = extract_engine_info(engine)
with st.expander("Hasil ekstraksi mesin"):
    st.write({
        "Horsepower": hp, "Displacement (L)": disp, "Cylinders": cyl,
        "Turbo": bool(turbo), "Konfigurasi": config,
    })

# ---------------- Prediksi ----------------
if st.button("Prediksi Harga", type="primary", use_container_width=True):
    if not str(model_name).strip():
        st.warning("Isi nama model terlebih dahulu.")
        st.stop()

    raw = {"brand": brand, "model": model_name, "model_year": model_year,
           "milage": mileage, "engine": engine}

    X = transform_one(raw, A)
    pred_log = model.predict(X)[0]          # target = log1p(price)
    price = float(np.expm1(pred_log))       # kembalikan ke USD

    st.success(f"### Estimasi harga: **$ {price:,.0f}**")
    st.caption(f"≈ Rp {price * 16_000:,.0f}  (kurs asumsi 16.000/USD)")

    with st.expander("Lihat 28 fitur yang dihitung"):
        st.dataframe(X.T.rename(columns={0: "nilai"}))

st.divider()
st.caption("Catatan: estimasi berbasis dataset training. Pastikan versi "
           "scikit-learn / catboost / lightgbm / xgboost sama dengan saat training.")
