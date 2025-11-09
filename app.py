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
st.header("📘 Project Overview")
st.markdown("""
This project explores how the **chemical properties** of red wine affect its **quality score**, based on data from the 
UCI Machine Learning Repository. Using **Multiple Linear Regression (MLR)**, we identify which variables most strongly 
influence the final rating — such as alcohol, acidity, sulphates, and density.

该项目基于葡萄酒化学特征，使用多元线性回归（MLR）模型预测红酒的质量评分。
通过建模与可视化，识别出影响酒质的关键因素，并验证模型假设的有效性。
""")

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

# -------------------- Visualization Gallery --------------------
st.markdown("""
Explore the model's diagnostic and exploratory visualizations below.
These charts illustrate correlations, distributions, and regression diagnostics 
to evaluate how chemical features influence predicted wine quality.
""")

import streamlit as st
from pathlib import Path

st.header("🖼️ Visualization Gallery")

# 所有可视化图像的路径和说明文字
image_info = [
    ("Assets/Correlation_Heatmap.png", "Correlation Heatmap — 变量间相关性可视化"),
    ("Assets/Pairwise_Relationships.png", "Pairwise Relationships — 成对特征关系"),
    ("Assets/Correlation_with_Wine_Quality.png", "Correlation with Wine Quality — 变量与酒质分数的线性关系"),
    ("Assets/Distribution_of_Wine_Features.png", "Distribution of Wine Features — 各特征分布情况"),
    ("Assets/Outliers_Detection.png", "Outliers Detection — 异常值检测结果"),
    ("Assets/Actual_vs_Predicted_Wine_Quality.png", "Actual vs Predicted Wine Quality — 预测结果对比"),
]

# 按三列展示
cols = st.columns(3)
for idx, (path, caption) in enumerate(image_info):
    col = cols[idx % 3]
    p = Path(path)
    if p.exists():
        col.image(str(p), caption=caption, use_container_width=True)
    else:
        col.error(f"⚠️ Missing: {p.name}")


# -------------------- Footer --------------------
st.markdown("""
<div class="footer">
Created by <b>Gia Hu</b> | Data from UCI ML Repository | Hosted on Streamlit Cloud
</div>
""", unsafe_allow_html=True)
