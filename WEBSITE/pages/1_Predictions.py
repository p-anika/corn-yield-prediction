import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="Predictions | CornCount", page_icon="🌽", layout="wide")

# 2. Custom CSS for Branding (Matches app.py)
st.markdown("""
    <style>
    /* Main background - Off-white */
    .stApp {
        background-color: #fdfdfb;
    }
    /* Metric and Container styling */
    [data-testid="stMetric"], .stElementContainer div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eeebe0;
    }
    .stMetric {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Branding
st.sidebar.markdown("# 🌽 CornCount")
st.sidebar.markdown("---")

# 4. Header
st.header("Predicting Corn Yield")
st.write("Add in data specific to your county's field to get your customized forecast")

# ── Model Loading (cached across reruns) ─────────────────────────────────────
_MODELS_BASE = Path(__file__).parent.parent / 'final_models'

@st.cache_resource(show_spinner="Loading CornCount models...")
def load_bundle():
    meta_path = _MODELS_BASE / 'metadata.joblib'
    if not meta_path.exists():
        return None, None
    from tensorflow.keras.models import load_model
    meta   = joblib.load(meta_path)
    models = {}
    for stage in ['V6', 'V12', 'R1', 'R3']:
        p = _MODELS_BASE / stage
        models[stage] = {
            'nn':   load_model(p / 'nn.keras'),
            'xgb':  joblib.load(p / 'xgb.joblib'),
            'lgbm': joblib.load(p / 'lgbm.joblib'),
        }
    return models, meta

models, meta = load_bundle()
MODELS_AVAILABLE = models is not None

if not MODELS_AVAILABLE:
    st.error("Saved models not found. Run the notebook's **Save All Models** cell first.")
    st.stop()

STAGES = ['V6', 'V12', 'R1', 'R3']
STAGE_LABELS = {
    'V6':  'V6  — ~6 leaves  (~June)',
    'V12': 'V12 — ~12 leaves (~July)',
    'R1':  'R1  — Silking    (~late July)',
    'R3':  'R3  — Milk stage (~August)',
}

# ── Helper: add bio features (mirrors notebook's add_bio_features) ─────────
def _add_bio(df, stage_lower, prev_stage=None):
    s = stage_lower
    df[f'{s}_thirst']    = df[f'{s}_vpd_mean'] / (df[f'{s}_precip_curr'] + 0.01)
    df[f'{s}_stress_idx'] = df[f'{s}_sdd_curr'] * df[f'{s}_vpd_mean']
    if prev_stage:
        df[f'{s}_velocity'] = df[f'{s}_evi_curr'] - df[f'{prev_stage}_evi_curr']
    else:
        df[f'{s}_velocity'] = df[f'{s}_evi_curr'] - df[f'{s}_evi_anom']
    return df

# ── Helper: assemble feature row and predict ─────────────────────────────────
def run_inference(row_dict, stage):
    sm       = meta['stages'][stage]
    feat_raw = sm['feature_names_raw']
    scaler_x = sm['scaler_x']
    scaler_y = sm['scaler_y']
    w        = sm['ensemble_weights']

    x_raw    = np.array([[row_dict.get(f, 0.0) for f in feat_raw]])
    x_scaled = scaler_x.transform(x_raw)

    # Compute bio features on scaled data (matches training procedure)
    df_scaled = pd.DataFrame(x_scaled, columns=feat_raw)
    df_eng    = _add_bio(df_scaled, sm['bio_stage_lower'], sm['bio_prev_stage'])
    x_eng     = df_eng.values

    mdl      = models[stage]
    xgb_s    = mdl['xgb'].predict(x_eng)[0]
    lgbm_s   = mdl['lgbm'].predict(x_eng)[0]
    try:
        nn_s = float(mdl['nn'].predict(x_eng, verbose=0)[0, 0])
    except Exception:
        # NN was trained on a pruned feature subset not saved to metadata; skip it.
        nn_s = None
    if nn_s is not None:
        blended = w['XGB'] * xgb_s + w['LGBM'] * lgbm_s + w['NN'] * nn_s
    else:
        total = w['XGB'] + w['LGBM']
        blended = (w['XGB'] * xgb_s + w['LGBM'] * lgbm_s) / total

    dummy         = np.zeros((1, 2))
    dummy[0, 0]   = blended
    pred_bushels  = float(scaler_y.inverse_transform(dummy)[0, 0])

    return pred_bushels, x_eng, df_eng.columns.tolist(), mdl, nn_s is not None

# ── Helper: compute weighted SHAP from XGB + LGBM ────────────────────────────
@st.cache_data(show_spinner=False)
def compute_shap(_xgb_model, _lgbm_model, x_eng_tuple, feat_names_tuple, w_xgb, w_lgbm):
    import shap
    x_eng      = np.array(x_eng_tuple).reshape(1, -1)
    feat_names = list(feat_names_tuple)
    sv_xgb  = shap.TreeExplainer(_xgb_model).shap_values(x_eng)[0]
    sv_lgbm = shap.TreeExplainer(_lgbm_model).shap_values(x_eng)[0]
    total   = w_xgb + w_lgbm
    sv      = (w_xgb * sv_xgb + w_lgbm * sv_lgbm) / total if total > 0 else sv_xgb
    return sv, feat_names

# ── Template CSV download ─────────────────────────────────────────────────────
def build_template_csv(stage):
    feat_raw = meta['stages'][stage]['feature_names_raw']
    return pd.DataFrame(columns=feat_raw).to_csv(index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1. Select Your Data Source")
data_mode = st.radio(
    "How should we gather feature data?",
    ["Public API Fetching (NASA/MODIS/NRCS)", "Manual CSV Upload"],
    help="Choose 'Public API' to let us pull data based on your coordinates, or 'Manual' if you have your own processed datasets."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 6. INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("2. Provide Location & Time")
c1, c2, c3, c4 = st.columns(4)
with c1:
    lat         = st.number_input("Latitude",  value=40.3387, format="%.4f")
with c2:
    lon         = st.number_input("Longitude", value=-88.9567, format="%.4f")
with c3:
    target_date = st.date_input("Until when do you have data?")
with c4:
    stage_sel   = st.selectbox(
        "Growth Stage",
        options=STAGES,
        format_func=lambda s: STAGE_LABELS[s],
        index=3,
        help="Select the latest growth stage for which you have data."
    )

feature_csv = None
if data_mode == "Manual CSV Upload":
    st.divider()
    st.subheader("3. Upload Feature File")
    st.info(
        "Upload a single-row CSV with all required features for your chosen stage. "
        "Download the template below to see the exact column names expected."
    )
    template_csv = build_template_csv(stage_sel)
    st.download_button(
        label=f"Download {stage_sel} feature template (.csv)",
        data=template_csv,
        file_name=f"corncount_{stage_sel.lower()}_template.csv",
        mime="text/csv",
    )
    feature_csv = st.file_uploader(
        "Upload completed feature CSV (one data row)",
        type="csv",
        key="features"
    )

st.write(" ")
submitted = st.button("Predict Yield", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. RESULTS SECTION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    if data_mode == "Manual CSV Upload" and feature_csv is None:
        st.error("Please upload a feature CSV before submitting.")
    elif data_mode == "Public API Fetching (NASA/MODIS/NRCS)":
        st.warning("Public API fetching is not yet implemented. Please use Manual CSV Upload.")
    else:
        # Parse uploaded CSV
        df_input = pd.read_csv(feature_csv)
        if df_input.empty:
            st.error("The uploaded CSV has no data rows.")
            st.stop()
        row_dict = df_input.iloc[0].to_dict()

        with st.status("Running CornCount ensemble...", expanded=True) as status:
            st.write(f"Scaling features for {stage_sel} stage...")
            pred_bushels, x_eng, feat_eng, mdl, nn_used = run_inference(row_dict, stage_sel)

            st.write("Computing SHAP feature importance...")
            w        = meta['stages'][stage_sel]['ensemble_weights']
            sv, feat_names = compute_shap(
                mdl['xgb'], mdl['lgbm'],
                tuple(x_eng.flatten()), tuple(feat_eng),
                w['XGB'], w['LGBM']
            )
            status.update(label="Prediction complete!", state="complete")

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            with st.container(border=True):
                st.markdown("### Estimated Yield")
                st.title(f"{pred_bushels:.1f} bu/ac")
                perf = meta['stages'][stage_sel]['performance']
                st.metric("Model MAE (test set)", f"± {perf['MAE']} bu/ac")
                st.write(f"**Coordinates:** {lat:.4f}, {lon:.4f}")
                st.write(f"**Phenological Stage:** {stage_sel}")
                wts = meta['stages'][stage_sel]['ensemble_weights']
                if nn_used:
                    st.caption(
                        f"Ensemble weights — XGB: {wts['XGB']:.2f} | "
                        f"LGBM: {wts['LGBM']:.2f} | NN: {wts['NN']:.2f}"
                    )
                else:
                    total = wts['XGB'] + wts['LGBM']
                    st.caption(
                        f"Ensemble weights — XGB: {wts['XGB']/total:.2f} | "
                        f"LGBM: {wts['LGBM']/total:.2f} | NN: unavailable "
                        f"(pruned feature list not saved; will be fixed on next retrain)"
                    )

        with res_col2:
            st.markdown("### Feature Influence (SHAP)")

            # Build top-10 SHAP bar chart
            shap_df = pd.DataFrame({'Feature': feat_names, 'SHAP': sv})
            shap_df['abs'] = shap_df['SHAP'].abs()
            shap_df = shap_df.nlargest(10, 'abs').sort_values('SHAP')
            shap_df['Color'] = shap_df['SHAP'].apply(lambda v: '#2e7d32' if v > 0 else '#c62828')

            fig = px.bar(
                shap_df, x='SHAP', y='Feature', orientation='h',
                color='Color', color_discrete_map='identity',
                labels={'SHAP': 'Impact on prediction (bu/ac, scaled)'}
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=360,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
