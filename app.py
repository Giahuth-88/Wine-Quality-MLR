import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Wine Quality MLR", page_icon="🍷", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #F8FAFC;
}
.main {
    padding: 0rem 2rem;
}
.big-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E293B;
}
.sub-title {
    font-size: 1.1rem;
    color: #475569;
}
.gradient-bg {
    background: linear-gradient(90deg, #e0f2fe 0%, #fdf2f8 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
}
.card {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(0,0,0,0.08);
}
.metric-card {
    background-color: #f9fafb;
    border-left: 5px solid #6366F1;
    padding: 1rem 1.5rem;
    border-radius: 8px;
}
.footer {
    font-size: 0.8rem;
    text-align: center;
    color: #6B7280;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Hero Section --------------------
st.markdown("""
<div class="gradient-bg">
    <h1 class="big-title">🍷 Wine Quality Prediction</h1>
    <p class="sub-title">Interactive machine learning demo using <b>Multiple Linear Regression</b> to analyze and predict wine quality based on chemical features.</p>
    <a href="https://github.com/Giahuth-88/Wine-Quality-MLR" target="_blank">
        <button style="background-color:#4F46E5;color:white;padding:0.5rem 1rem;border:none;border-radius:6px;cursor:pointer;margin-top:0.5rem;">View on GitHub</button>
    </a>
</div>
""", unsafe_allow_html=True)

# -------------------- Project Overview --------------------
st.header("📘 Project Background / 项目背景")
st.markdown("""
**Goal.** Predict red wine quality scores from chemistry using **Multiple Linear Regression (MLR)** and
identify the most influential factors that drive perceived quality.

**Dataset.** UCI Machine Learning Repository – *Wine Quality (Red)*, 1,599 samples with 11 chemistry features
(e.g., alcohol, volatile acidity, citric acid, sulphates, density) and a sensory **quality score (0–10)**.

**Method.** Classic OLS via `statsmodels`:
- Checked multicollinearity (VIF < 5 for all features)
- Residual diagnostics (Residuals vs Fitted, QQ plot)
- Coefficient interpretation with confidence bands

**Key findings.**
- **Alcohol** and **sulphates** show **positive** association with quality  
- **Volatile acidity** shows a **negative** association  
- Model goodness-of-fit **R² ≈ {:.3f}** (reasonable for human-scored sensory data)

**How to use this app.**  
Use the sliders on the left to simulate chemistry values. The app will return the predicted quality with a 95% CI, and show model diagnostics below.

**项目简介（中文）**  
本项目基于 UCI 红酒质量数据集，使用多元线性回归预测酒质评分，并通过可视化解释关键影响因素。
结果表明：**酒精度**与**硫酸盐**与质量呈正相关，**挥发性酸度**与质量呈负相关，模型拟合优度 **R² ≈ {:.3f}**。左侧滑块可交互模拟不同化学参数对评分的影响。
""".format(model.rsquared))

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/winequality-red.csv")
    cols = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density", "quality"]
    df = df.rename(columns=lambda c: c.strip())
    return df[cols]

@st.cache_resource
def fit_model(df):
    X = df[["alcohol", "volatile acidity", "sulphates", "citric acid", "density"]]
    X = sm.add_constant(X)
    y = df["quality"]
    model = sm.OLS(y, X).fit()
    return model

df = load_data()
model = fit_model(df)

# -------------------- Key Results Section --------------------
st.header("📊 Model Performance Summary")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <b>R² (Goodness of Fit):</b> {model.rsquared:.3f} <br>
        <b>Mean Quality:</b> {df['quality'].mean():.2f} <br>
        <b>Sample Size:</b> {len(df)} wines
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <b>Most influential factors:</b><br>
        ✅ Alcohol (positive)<br>
        ❌ Volatile acidity (negative)<br>
        ✅ Sulphates (positive)
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -------------------- Interactive Prediction --------------------
st.header("🔮 Interactive Model Explorer")

st.sidebar.header("Input Features")

def slider(label, s, e, v, step=0.01):
    return st.sidebar.slider(label, float(s), float(e), float(v), step=step)

q = df.quantile([0.05, 0.5, 0.95])
alcohol = slider("Alcohol (%)", q.loc[0.05,"alcohol"], q.loc[0.95,"alcohol"], q.loc[0.5,"alcohol"])
volatile = slider("Volatile acidity (g/dm³)", q.loc[0.05,"volatile acidity"], q.loc[0.95,"volatile acidity"], q.loc[0.5,"volatile acidity"])
sulphates = slider("Sulphates (g/dm³)", q.loc[0.05,"sulphates"], q.loc[0.95,"sulphates"], q.loc[0.5,"sulphates"])
citric = slider("Citric acid (g/dm³)", q.loc[0.05,"citric acid"], q.loc[0.95,"citric acid"], q.loc[0.5,"citric acid"])
density = slider("Density (g/cm³)", q.loc[0.05,"density"], q.loc[0.95,"density"], q.loc[0.5,"density"], step=0.0001)

X_new = pd.DataFrame({
    "const":[1.0],
    "alcohol":[alcohol],
    "volatile acidity":[volatile],
    "sulphates":[sulphates],
    "citric acid":[citric],
    "density":[density],
})
pred = model.get_prediction(X_new)
res = pred.summary_frame(alpha=0.05)

left, right = st.columns(2)
with left:
    st.subheader("Predicted Quality")
    st.metric("Predicted Score", f"{res['mean'].iloc[0]:.2f}")
    st.caption(f"95% CI: {res['obs_ci_lower'].iloc[0]:.2f} — {res['obs_ci_upper'].iloc[0]:.2f}")
with right:
    st.subheader("Model Summary")
    st.write(f"**R²:** {model.rsquared:.3f}")
    st.write("Significant variables (p < 0.05):")
    sig = model.pvalues[model.pvalues<0.05].index.tolist()
    st.write(", ".join([s for s in sig if s!='const']) or "None")

st.divider()

# -------------------- Visualization Section --------------------
st.header("📈 Data Visualization")

from pathlib import Path

col1, col2 = st.columns(2)

img1 = Path("Assets/Correlation_Heatmap.png")
img2 = Path("Assets/Pairwise_Relationships.png")

def show(p: Path, holder, caption: str):
    if p.exists():
        holder.image(str(p), caption=caption, use_container_width=True)
    else:
        st.error(f"Image not found: {p} (check file name & path)")

show(img1, col1, "Correlation Heatmap")
show(img2, col2, "Pairwise Relationships")

# -------------------- Footer --------------------
st.markdown("""
<div class="footer">
Created by <b>Gia Hu</b> | Data from UCI ML Repository | Hosted on Streamlit Cloud
</div>
""", unsafe_allow_html=True)
