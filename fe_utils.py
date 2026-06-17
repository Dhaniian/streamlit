"""
fe_utils.py
-----------
Feature engineering untuk INFERENSI (1 input baru) — mereplikasi persis
fungsi create_28_features() di ProjekFinal.ipynb.

Dipakai oleh:
  - notebook  : build_fe_artifacts(df_train_raw)  -> simpan artefak
  - app.py    : transform_one(raw_input, artifacts) -> 28 fitur siap predict

PENTING: 28 fitur dibangun dari statistik DATA TRAIN (brand/model frequency,
median imputasi, KMeans, scaler, cluster encoding). Statistik ini WAJIB
disimpan saat training dan dipakai ulang saat inferensi agar konsisten.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CURRENT_YEAR = 2024

# Urutan 28 fitur — HARUS sama persis dengan notebook
FEATURE_NAMES = [
    'brand_frequency', 'age_mileage_product', 'engine_config_encoded', 'engine_config_Unknown',
    'cluster', 'displacement_per_cylinder', 'engine_config_Rotary', 'power_per_cylinder',
    'engine_config_Inline', 'cluster_encoded', 'age_mileage_interaction', 'efficiency_score',
    'brand_count', 'power_to_weight_estimate', 'engine_displacement', 'model_frequency',
    'engine_config_V-Type', 'cylinders', 'has_turbo', 'luxury_age_interaction',
    'power_per_liter_log', 'power_age_interaction', 'power_per_liter', 'brand_reliability_score',
    'horsepower', 'model_count', 'performance_age_interaction', 'horsepower_squared'
]

CLUSTER_FEATURES = ['horsepower', 'engine_displacement', 'mileage', 'age']
CONFIG_ORDER = {'Unknown': 0, 'Inline': 1, 'V-Type': 2, 'Rotary': 3}
RELIABLE_BRANDS = ['Toyota', 'Honda', 'Mazda', 'Lexus', 'Acura', 'Subaru']
LUXURY_BRANDS = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'Jaguar',
                 'Land Rover', 'Cadillac', 'Tesla', 'Maserati', 'Bentley',
                 'Rolls-Royce', 'Ferrari', 'Lamborghini', 'Aston Martin']


# ---------------------------------------------------------------------------
# 1) Parsing engine string  (identik dengan notebook)
# ---------------------------------------------------------------------------
def extract_engine_info(engine_str):
    """Return: horsepower, displacement, cylinders, has_turbo, engine_config."""
    if engine_str is None or (isinstance(engine_str, float) and pd.isna(engine_str)):
        return None, None, None, 0, 'Unknown'

    engine_lower = str(engine_str).lower()

    hp = None
    m = re.search(r'(\d+\.?\d*)\s*hp', engine_lower)
    if m:
        hp = float(m.group(1))
        if not (50 <= hp <= 2000):
            hp = None

    disp = None
    m = re.search(r'(\d+\.?\d*)\s*l', engine_lower)
    if m:
        disp = float(m.group(1))
        if not (0.5 <= disp <= 8.5):
            disp = None

    cylinders = None
    m = re.search(r'v(\d+)|i(\d+)|(\d+)\s*cyl', engine_lower)
    if m:
        for g in m.groups():
            if g:
                cylinders = int(g)
                break

    turbo = 1 if any(x in engine_lower for x in
                     ['turbo', 'turbocharged', 'tsi', 'supercharge']) else 0

    if any(v in engine_lower for v in ['v6', 'v8', 'v10', 'v12']):
        config = 'V-Type'
    elif any(v in engine_lower for v in ['i4', 'i6', 'inline', 'straight']):
        config = 'Inline'
    elif 'rotary' in engine_lower or 'wankel' in engine_lower:
        config = 'Rotary'
    else:
        config = 'Unknown'

    return hp, disp, cylinders, turbo, config


def _parse_basic(df):
    df = df.copy()
    df['price'] = (df['price'].astype(str)
                   .str.replace('$', '', regex=False)
                   .str.replace(',', '', regex=False))
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['mileage'] = df['milage'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    for col in ['fuel_type', 'accident', 'clean_title']:
        if col in df.columns:
            df[col] = df[col].replace(['–', '', 'not supported'], np.nan)
    return df


# ---------------------------------------------------------------------------
# 2) Bangun artefak dari DATA TRAIN  (dijalankan SEKALI di notebook)
# ---------------------------------------------------------------------------
def build_fe_artifacts(df_train_raw):
    """
    Hitung ulang seluruh statistik train (deterministik, random_state=42)
    dan kembalikan dict artefak untuk inferensi. Tidak perlu Optuna ulang.
    """
    df = _parse_basic(df_train_raw)

    # Outlier filter (threshold dari train) — sama seperti notebook
    q_low, q_high = df['price'].quantile(0.005), df['price'].quantile(0.995)
    df = df[(df['price'] >= q_low) & (df['price'] <= q_high)].copy()
    df = df[df['price'] > 0].dropna(subset=['price'])

    # age + engine info
    df['age'] = (CURRENT_YEAR - df['model_year']).clip(lower=0)
    eng = df['engine'].apply(extract_engine_info)
    df['horsepower'] = eng.apply(lambda x: x[0])
    df['engine_displacement'] = eng.apply(lambda x: x[1])
    df['cylinders'] = eng.apply(lambda x: x[2])
    df['has_turbo'] = eng.apply(lambda x: x[3])
    df['engine_config'] = eng.apply(lambda x: x[4])

    # Imputasi median (global / brand / brand+model)
    global_med, brand_med, bm_med = {}, {}, {}
    for col in ['horsepower', 'engine_displacement', 'cylinders']:
        global_med[col] = df[col].median()
        brand_med[col] = df.groupby('brand')[col].median().to_dict()
        bm_med[col] = df.groupby(['brand', 'model'])[col].median().to_dict()

        def fill(row, c=col):
            if pd.notna(row[c]):
                return row[c]
            v = bm_med[c].get((row['brand'], row['model']))
            if v is not None and not pd.isna(v):
                return v
            v = brand_med[c].get(row['brand'])
            if v is not None and not pd.isna(v):
                return v
            return global_med[c]

        df[col] = df.apply(fill, axis=1)
        df[col] = df[col].fillna(df[col].median())

    # Brand / model market maps
    brand_freq = df['brand'].value_counts(normalize=True).to_dict()
    brand_count = df['brand'].value_counts().to_dict()
    model_freq = df['model'].value_counts(normalize=True).to_dict()
    model_count = df['model'].value_counts().to_dict()

    # Clustering (fit di train)
    Xc = df[CLUSTER_FEATURES].copy()
    cluster_feature_medians = {c: Xc[c].median() for c in CLUSTER_FEATURES}
    for c in CLUSTER_FEATURES:
        Xc[c] = Xc[c].fillna(cluster_feature_medians[c])

    scaler_cluster = StandardScaler()
    Xc_scaled = scaler_cluster.fit_transform(Xc)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(Xc_scaled)

    global_mean_price = np.log1p(df['price']).mean()
    stats = df.groupby('cluster').agg({'price': ['mean', 'count']})
    stats.columns = ['mean', 'count']
    smoothing = 10
    cluster_encoded = ((np.log1p(stats['mean']) * stats['count']
                        + global_mean_price * smoothing)
                       / (stats['count'] + smoothing)).to_dict()

    # Median fitur final (untuk fillna 28 fitur) — bangun matriks train
    artifacts = dict(
        feature_names=FEATURE_NAMES, cluster_features=CLUSTER_FEATURES,
        config_order=CONFIG_ORDER, reliable_brands=RELIABLE_BRANDS,
        luxury_brands=LUXURY_BRANDS, current_year=CURRENT_YEAR,
        global_med=global_med, brand_med=brand_med, bm_med=bm_med,
        brand_freq=brand_freq, brand_count=brand_count,
        model_freq=model_freq, model_count=model_count,
        scaler_cluster=scaler_cluster, kmeans=kmeans,
        cluster_encoded=cluster_encoded, global_mean_price=global_mean_price,
        cluster_feature_medians=cluster_feature_medians,
        feature_medians={f: 0.0 for f in FEATURE_NAMES},  # diisi di bawah
        brands=sorted([b for b in brand_freq.keys() if isinstance(b, str)]),
        models_by_brand=(df.groupby('brand')['model']
                         .apply(lambda s: sorted(s.dropna().unique().tolist()))
                         .to_dict()),
    )

    # Hitung median 28 fitur dari seluruh baris train
    rows = [transform_one(r, artifacts, _fillna=False)
            for r in df[['brand', 'model', 'model_year', 'milage', 'engine']]
            .to_dict('records')]
    Xtr = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    artifacts['feature_medians'] = {f: float(Xtr[f].median()) for f in FEATURE_NAMES}

    return artifacts


# ---------------------------------------------------------------------------
# 3) Transform 1 input mentah -> 28 fitur  (dipakai app.py)
# ---------------------------------------------------------------------------
def transform_one(raw, A, _fillna=True):
    """
    raw : dict berisi 'brand', 'model', 'model_year', 'milage' (atau 'mileage'), 'engine'
    A   : artifacts dari build_fe_artifacts
    return: DataFrame 1 baris, 28 kolom (urutan FEATURE_NAMES)
    """
    cy = A['current_year']
    brand = raw.get('brand')
    model = raw.get('model')
    model_year = float(raw.get('model_year'))

    mil_raw = re.sub(r'[^0-9]', '', str(raw.get('milage', raw.get('mileage', ''))))
    mileage = pd.to_numeric(mil_raw, errors='coerce')
    if pd.isna(mileage):
        mileage = A['cluster_feature_medians']['mileage']

    age = max(cy - model_year, 0)
    hp, disp, cyl, turbo, config = extract_engine_info(raw.get('engine'))

    def impute(col, val):
        if val is not None and not pd.isna(val):
            return float(val)
        v = A['bm_med'][col].get((brand, model))
        if v is not None and not pd.isna(v):
            return float(v)
        v = A['brand_med'][col].get(brand)
        if v is not None and not pd.isna(v):
            return float(v)
        return float(A['global_med'][col])

    horsepower = impute('horsepower', hp)
    engine_displacement = max(impute('engine_displacement', disp), 0.5)
    cylinders = max(impute('cylinders', cyl), 1)

    f = {}
    f['horsepower'] = horsepower
    f['engine_displacement'] = engine_displacement
    f['cylinders'] = cylinders
    f['has_turbo'] = turbo
    f['horsepower_squared'] = horsepower ** 2

    # Interactions
    f['age_mileage_interaction'] = age * mileage / 10000
    f['age_mileage_product'] = np.log1p(age + 1) * np.log1p(mileage / 1000)
    f['power_age_interaction'] = horsepower * (1 / (age + 1))
    f['performance_age_interaction'] = horsepower / (age + 1)

    # Engine performance
    f['power_per_liter'] = horsepower / engine_displacement
    f['power_per_liter_log'] = np.log1p(f['power_per_liter'])
    f['power_per_cylinder'] = horsepower / cylinders
    f['displacement_per_cylinder'] = engine_displacement / cylinders
    f['efficiency_score'] = horsepower / (engine_displacement * cylinders)
    f['power_to_weight_estimate'] = horsepower / (horsepower * 30 / 2000 + 3)

    # Engine config
    f['engine_config_encoded'] = A['config_order'].get(config, 0)
    f['engine_config_Unknown'] = int(config == 'Unknown')
    f['engine_config_Inline'] = int(config == 'Inline')
    f['engine_config_V-Type'] = int(config == 'V-Type')
    f['engine_config_Rotary'] = int(config == 'Rotary')

    # Brand / model market
    f['brand_frequency'] = A['brand_freq'].get(brand, 0)
    f['brand_count'] = A['brand_count'].get(brand, 1)
    f['model_frequency'] = A['model_freq'].get(model, 0)
    f['model_count'] = A['model_count'].get(model, 1)
    f['brand_reliability_score'] = 1.0 if brand in A['reliable_brands'] else 0.5
    is_luxury = int(brand in A['luxury_brands'])
    f['luxury_age_interaction'] = is_luxury * age

    # Clustering
    cf = pd.DataFrame([[horsepower, engine_displacement, mileage, age]],
                      columns=A['cluster_features'])
    cluster = int(A['kmeans'].predict(A['scaler_cluster'].transform(cf))[0])
    f['cluster'] = cluster
    f['cluster_encoded'] = A['cluster_encoded'].get(cluster, A['global_mean_price'])

    row = pd.DataFrame([[f[name] for name in A['feature_names']]],
                       columns=A['feature_names'])
    row = row.replace([np.inf, -np.inf], np.nan)
    if _fillna:
        for c in A['feature_names']:
            if row[c].isna().any():
                row[c] = row[c].fillna(A['feature_medians'][c])
    return row
