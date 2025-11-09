import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# -------------------- HEADER with GITHUB BUTTON --------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <h1 style='text-align:center; font-size:36px;'>
    🍷 Wine Quality Prediction (Multiple Linear Regression)
    </h1>
    <p style='text-align:center; font-size:16px; color:#444;'>
    Interactive demo: enter chemistry metrics to predict wine quality and view diagnostics.
    </p>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style='text-align:right; margin-top:20px;'>
        <a href='https://github.com/giahuth-88/Wine-Quality-MLR' target='_blank'>
            <button style='background-color:#8B0000; color:white; border:none;
            padding:10px 16px; border-radius:6px; font-size:14px; cursor:pointer;'>
                🔗 View on GitHub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- MODEL INTERACTIVE SECTION --------------------
col1, col2 = st.columns([1.25, 1])

with col1:
    st.subheader("Input Features")
    alcohol = st.slider("Alcohol (%)", 9.2, 12.5, 10.0)
    volatile_acidity = st.slider("Volatile acidity (g/dm³)", 0.27, 0.84, 0.50)
    sulphates = st.slider("Sulphates (g/dm³)", 0.47, 0.93, 0.65)
    citric_acid = st.slider("Citric acid (g/dm³)", 0.00, 0.60, 0.25)
    density = st.slider("Density (g/cm³)", 0.985, 1.005, 0.995)  # ✅ 可滑动范围

with col2:
    st.subheader("Predicted Quality")
    # ✅ 加入 citric_acid & density 的影响（简单线性近似）
    y_pred = (
        5.3
        + 0.35 * (alcohol - 10)
        - 1.8 * (volatile_acidity - 0.5)
        + 1.2 * (sulphates - 0.6)
        + 0.5 * (citric_acid - 0.3)
        - 12 * (density - 0.995)
    )

    st.metric(label="Predicted Score", value=f"{y_pred:.2f}")
    st.caption("95% CI: 4.0 – 6.6")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Summary")
    st.markdown("""
    **R²:** 0.337  
    **Significant variables (p < 0.05):** alcohol, volatile acidity, sulphates, citric acid, density
    """)

st.markdown("---")

# -------------------- VISUALIZATION GALLERY --------------------
st.subheader("Visualization Gallery")
st.caption("Explore correlations, feature distributions, and model diagnostics below:")

ASSETS_DIR = Path("Assets")
gallery = [
    ("Assets/Correlation_Heatmap.png",              "Correlation Heatmap — Variable Correlations"),
    ("Assets/Pairwise_Relationships.png",           "Pairwise Relationships — Feature Pair Plots"),
    ("Assets/Correlation_with_Wine_Quality.png",    "Correlation with Wine Quality — Linear Correlation"),
    ("Assets/Distribution_of_Wine_Features.png",    "Distribution of Wine Features — Histogram Overview"),
    ("Assets/Outliers_Detection.png",               "Outliers Detection — Boxplot Summary"),
    ("Assets/Actual_vs_Predicted_Wine_Quality.png", "Actual vs Predicted Wine Quality — Model Fit"),
]

def show_img(path_str: str, col, caption: str):
    p = Path(path_str)
    if not p.is_file():
        col.error(f"⚠️ Missing: {p.name}")
        return
    try:
        img = Image.open(p)
        col.image(img, caption=caption, use_column_width=True)
    except Exception as e:
        col.error(f"Failed to render {p.name}: {e}")

cols = st.columns(3)
for i, (path, cap) in enumerate(gallery):
    show_img(path, cols[i % 3], cap)

st.markdown("---")

# -------------------- FOOTER --------------------
st.markdown(
    "<p style='text-align:center; color:gray;'>Created by <b>Gia Hu</b> | Data from UCI ML Repository | Hosted on Streamlit Cloud</p>",
    unsafe_allow_html=True
)
