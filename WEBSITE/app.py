import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="CornCount", page_icon="🌽", layout="wide")

# 2. Styling (Off-White Background & Custom Branding)
def apply_custom_styles():
    st.markdown("""
        <style>
        /* Main background - Off-white */
        .stApp {
            background-color: #fdfdfb;
        }
        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #eeebe0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .corn-header {
            color: #2e7d32;
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 0px;
        }
        </style>
        """, unsafe_allow_html=True)

apply_custom_styles()

# 3. Sidebar Branding
with st.sidebar:
    st.markdown("# 🌽 CornCount")
    st.divider()
    st.info("Working towards earlier, more accurate yield forecasts.")
    st.success("Click 'Predictions' to start.")

# 4. Main Content
st.markdown("<div class='corn-header'>CornCount</div>", unsafe_allow_html=True)
st.markdown("### *A Phenologically-Aware Machine Learning Framework*")
st.divider()

# 5. Key Statistics Section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("In 2023, weather and fire events caused losses of", "$20.3 billion")

with col2:
    st.metric("Corn faced the highest impact at", "$3.85 billion")

with col3:
    st.metric("Out of all U.S. crop commodities, corn is the", "2nd most valued")

st.divider()

# 6. Project Narrative
st.header("About the Project")
st.write("""
CornCount is a neural network and XGBoost ensemble model that forecasts corn yield by
taking advantage of the crop’s different sensitivity to external conditions at various
stages in the growing season. It uses multi-modal data and biologically-meaningful feature
engineering. It provides earlier forecasting and aims to increase forecast accuracy
amidst increasing climate volatility.
""")

st.subheader("Data Used")
st.markdown("""
- **USDA NASS**: Used for the ground truth county-level corn yield for training the model and for historical average input features.
- **NASA POWER**: Provided weather data for the counties. It had precipitation, maximum and minimum temperature, and more.
- **NRCS Soil Data Access**: Gave soil data for counties, including the sand/clay/silt composition, organic matter, and carbon exchange capacity.
- **MODIS Data**: Gave satellite data to calculate vegetation indices (EVI and NDWI).
""")