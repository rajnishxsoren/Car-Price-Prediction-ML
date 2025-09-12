# app_pretty_minchange.py — Same UI as v2.1, only add dependent model filtering
import io
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup & theme helpers
# -----------------------------
st.set_page_config(page_title="✨ Car Price Predictor", page_icon="🚗", layout="wide")

def inject_css(accent: str = "#7c3aed"):
    st.markdown(f"""
    <style>
    :root {{ --accent: {accent}; }}
    .app-header {{ display:flex; align-items:center; gap:14px; margin: 6px 0 12px 0; }}
    .app-badge {{
      padding: 6px 10px; border-radius: 999px; background: rgba(124,58,237,0.10);
      color: var(--accent); font-weight: 600; border: 1px solid rgba(124,58,237,0.2);
    }}
    .gradient-title {{
      font-weight: 800; font-size: 28px;
      background: linear-gradient(90deg, var(--accent), #06b6d4);
      -webkit-background-clip: text; background-clip: text; color: transparent; margin: 0;
    }}
    .sub {{ color: rgba(0,0,0,0.6); }}
    .stats-row {{ display:grid; grid-template-columns: 1.2fr 2fr 0.8fr; gap: 16px; }}
    .stat-card {{
      border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px 16px;
      background: rgba(255,255,255,0.7); backdrop-filter: blur(8px);
    }}
    .stat-label {{ font-size: 12px; opacity: .7; margin-bottom: 6px; }}
    .stat-value {{ font-size: 20px; font-weight: 800; line-height: 1.25; }}
    .filename {{ color: #f59e0b; word-break: break-word; }}
    .features {{ color: #22c55e; white-space: normal; word-wrap: break-word; overflow-wrap: anywhere; }}
    .rows {{ color: #f59e0b; }}
    @media (prefers-color-scheme: dark) {{
      .sub {{ color: rgba(255,255,255,0.7);}}
      .stat-card {{ background: rgba(22,22,24,0.5); border-color: rgba(255,255,255,0.08);}}
    }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def joblib_then_pickle_load(raw: bytes):
    try:
        import joblib
        return joblib.load(io.BytesIO(raw)), "joblib"
    except Exception:
        return pickle.loads(raw), "pickle"

@st.cache_resource(show_spinner=False)
def load_model_bytes(raw: bytes, filename: str):
    model, loader = joblib_then_pickle_load(raw)
    return model, loader

@st.cache_resource(show_spinner=False)
def load_model_from_file(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    return load_model_bytes(raw, path)

@st.cache_data(show_spinner=False)
def load_csv(path_or_file) -> pd.DataFrame:
    if hasattr(path_or_file, "read"):
        return pd.read_csv(path_or_file)
    return pd.read_csv(path_or_file)

def infer_expected_columns(model) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        try:
            return list(map(str, model.feature_names_in_))
        except Exception:
            pass
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline):
            for _, step in model.steps:
                if isinstance(step, ColumnTransformer):
                    cols = []
                    for _, _, c in step.transformers:
                        if c is None: continue
                        if isinstance(c, (list, tuple, np.ndarray, pd.Index)):
                            cols.extend([str(x) for x in c])
                        elif isinstance(c, str):
                            cols.append(c)
                    if cols:
                        seen=set(); ordered=[]
                        for x in cols:
                            if x not in seen: seen.add(x); ordered.append(x)
                        return ordered
    except Exception:
        pass
    return ['name','company','year','kms_driven','fuel_type']

def predict_df(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df)

def meta_for_model(model) -> Dict[str, object]:
    out = {"type": f"{type(model).__module__}.{type(model).__name__}"}
    for attr in ["n_features_in_", "feature_names_in_", "classes_"]:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                out[attr] = val
            except Exception:
                pass
    for attr in ["coef_", "intercept_"]:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                out[attr] = val
            except Exception:
                pass
    return out

# -----------------------------
# Sidebar (same as v2.1)
# -----------------------------
MODEL_DEFAULT = "LinearRegressionModel.pkl"
DATA_DEFAULT = "Cleaned_Car_data.csv"

with st.sidebar:
    st.header("⚙️ Setup")
    accent = st.selectbox("Accent color", ["#7c3aed", "#ef4444", "#22c55e", "#3b82f6", "#f59e0b"], index=0)
    inject_css(accent)

    up_model = st.file_uploader("Upload model (.pkl/.joblib)", type=["pkl","joblib"])
    up_csv = st.file_uploader("Upload cleaned CSV", type=["csv"])

    model = None; model_origin = None; model_filename = None
    if up_model is not None:
        model, loader = load_model_bytes(up_model.read(), up_model.name)
        model_origin = f"Uploaded ({loader})"
        model_filename = up_model.name
        st.success("Model loaded from upload.")
    else:
        try:
            model, loader = load_model_from_file(MODEL_DEFAULT)
            model_origin = f"Local file ({loader})"
            model_filename = os.path.basename(MODEL_DEFAULT)
            st.info("Model loaded from local file.")
        except Exception:
            st.error("No model loaded. Upload a model or place 'LinearRegressionModel.pkl' next to the app.")
            st.stop()

    car_df = None
    if up_csv is not None:
        car_df = load_csv(up_csv)
        st.success("CSV loaded from upload.")
    else:
        try:
            car_df = load_csv(DATA_DEFAULT)
            st.caption("Using local Cleaned_Car_data.csv")
        except Exception:
            st.caption("No local CSV found (dropdowns will use defaults).")

# -----------------------------
# Header (same)
# -----------------------------
st.markdown('<div class="app-header"><span class="app-badge">v2.1 • Streamlit</span><h1 class="gradient-title">Car Price Predictor</h1></div>', unsafe_allow_html=True)
st.write('<span class="sub">Sleek UI • Upload model/CSV • Single & batch predictions • Insights • History</span>', unsafe_allow_html=True)

# -----------------------------
# Stats row (same)
# -----------------------------
exp_cols = infer_expected_columns(model)
rows = car_df.shape[0] if isinstance(car_df, pd.DataFrame) else 0

st.markdown(
    f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-label">Model</div>
        <div class="stat-value"><span class="filename">{model_filename}</span></div>
        <div class="sub">{model_origin}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Expected features</div>
        <div class="stat-value features">{', '.join(map(str, exp_cols))}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Rows in CSV</div>
        <div class="stat-value rows">{rows:,}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Tabs (same labels)
# -----------------------------
tab_predict, tab_batch, tab_insights, tab_history = st.tabs(["🔮 Predict", "📦 Batch", "📈 Insights", "🕒 History"])

# -----------------------------
# Predict tab — MINIMAL CHANGE: filter car models by selected company
# -----------------------------
with tab_predict:
    st.subheader("Single Prediction")
    if car_df is not None:
        # Build lists exactly as before
        companies = sorted(car_df['company'].dropna().unique().tolist())
        companies.insert(0, 'Select Company')
        years = sorted(car_df['year'].dropna().unique().tolist(), reverse=True)
        fuel_types = car_df['fuel_type'].dropna().unique().tolist()

        # NEW: normalized helper for filtering (no visual change)
        df_norm = car_df.copy()
        for col in ['company','name']:
            if col in df_norm.columns:
                df_norm[col] = df_norm[col].astype(str).str.strip()

        # Selected company
        c1, c2 = st.columns([1,1])
        with c1:
            company = st.selectbox("Company", companies, index=0)
            year = st.selectbox("Year", years)
            kms = st.number_input("Kilometers Driven", min_value=0, step=500, value=0, key="kms_single")

        with c2:
            # MINIMAL CHANGE: restrict models to the chosen company
            if company and company != 'Select Company':
                models = sorted(
                    df_norm.loc[df_norm['company'] == str(company).strip(), 'name']
                    .dropna().unique().tolist()
                )
            else:
                models = sorted(car_df['name'].dropna().unique().tolist())

            # keep previous selection if still valid
            prev = st.session_state.get("car_model_prev")
            idx = models.index(prev) if (prev in models) else 0
            car_model = st.selectbox("Car Model (name)", options=models, index=idx)
            st.session_state["car_model_prev"] = car_model

            fuel_type = st.selectbox("Fuel Type", options=fuel_types)

    else:
        # Fallback values (unchanged)
        companies = ["Select Company","Maruti","Hyundai","Tata","Honda","Mahindra"]
        car_models = ["Alto","Swift","i20","City","Nexon"]
        years = list(range(2025, 1995, -1))
        fuel_types = ["Petrol","Diesel","CNG","LPG"]

        c1, c2 = st.columns([1,1])
        with c1:
            company = st.selectbox("Company", companies, index=0)
            year = st.selectbox("Year", years)
            kms = st.number_input("Kilometers Driven", min_value=0, step=500, value=0, key="kms_single")
        with c2:
            car_model = st.selectbox("Car Model (name)", car_models)
            fuel_type = st.selectbox("Fuel Type", fuel_types)

    if st.button("✨ Predict Price", type="primary"):
        try:
            X = pd.DataFrame(
                columns=['name','company','year','kms_driven','fuel_type'],
                data=np.array([car_model, company, year, kms, fuel_type]).reshape(1,5)
            )
            y = predict_df(model, X)
            price = float(np.array(y).ravel()[0])
            c1, c2 = st.columns([1,2])
            with c1: st.metric("Estimated Price", f"₹ {price:,.2f}")
            with c2:
                st.markdown(
                    f"""<div class="stat-card">
                        <span class="sub">Selection</span><br>
                        <span class="features">{company} • {car_model} • {int(year)} • {fuel_type} • {int(kms)} km</span>
                    </div>""",
                    unsafe_allow_html=True
                )
            if "history" not in st.session_state: st.session_state.history = []
            st.session_state.history.append({
                "name": car_model, "company": company, "year": int(year),
                "kms_driven": int(kms), "fuel_type": fuel_type, "prediction": round(price,2)
            })
        except Exception as e:
            st.error("Prediction failed. Ensure your model expects ['name','company','year','kms_driven','fuel_type'].")
            st.exception(e)

# -----------------------------
# Batch / Insights / History (unchanged)
# -----------------------------
with tab_batch:
    st.subheader("Batch Predictions")
    st.caption("Upload a CSV with columns ['name','company','year','kms_driven','fuel_type'].")
    uploaded_csv = st.file_uploader("Upload CSV for batch", type=["csv"], key="batch_csv")
    if uploaded_csv is not None:
        try:
            dfb = pd.read_csv(uploaded_csv)
            st.write("Preview:"); st.dataframe(dfb.head(20))
            needed = ['name','company','year','kms_driven','fuel_type']
            if all(c in dfb.columns for c in needed):
                X = dfb[needed].copy()
                X['year'] = pd.to_numeric(X['year'], errors='coerce')
                X['kms_driven'] = pd.to_numeric(X['kms_driven'], errors='coerce')
                pred = predict_df(model, X)
                out = dfb.copy(); out['prediction'] = pred
                st.success("Predictions generated."); st.dataframe(out.head(50))
                st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            else:
                st.error(f"CSV must contain columns: {needed}")
        except Exception as e:
            st.error("Failed to read CSV or predict."); st.exception(e)

with tab_insights:
    st.subheader("Model Insights")
    st.json(meta_for_model(model))

with tab_history:
    st.subheader("History (this session)")
    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        hdf = pd.DataFrame(st.session_state.history); st.dataframe(hdf)
        st.download_button("⬇️ Download history", hdf.to_csv(index=False).encode("utf-8"), "history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []; st.experimental_rerun()
