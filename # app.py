# app.py
# Customer Segmentation App — Streamlit + scikit-learn
# Run: streamlit run app.py

import io
import base64
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

import plotly.express as px

# ---------------------- UI SETUP ----------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧭 Customer Segmentation (Unsupervised ML)")
st.caption(
    "Upload your customer CSV, pick features, choose an algorithm, and segment your customers.")

with st.sidebar:
    st.header("1) Data Input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    demo = st.checkbox("Use demo synthetic data", value=uploaded is None)

# ---------------------- DATA LOADING ----------------------


def load_demo_data(n=400, random_state=42):
    rng = np.random.default_rng(random_state)
    # Create synthetic segments: Value Shoppers, Premium Loyal, Deal Seekers, New/Cold
    seg = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.25, 0.30, 0.15])

    income = (seg == 1)*rng.normal(90000, 12000, n) + \
        (seg != 1)*rng.normal(45000, 15000, n)
    recency_days = (seg == 2)*rng.normal(10, 8, n) + (seg != 2) * \
        rng.normal(45, 20, n)  # lower is more recent
    frequency = (seg == 1)*rng.normal(22, 6, n) + \
        (seg != 1)*rng.normal(8, 5, n)
    monetary = (seg == 1)*rng.normal(1200, 300, n) + \
        (seg != 1)*rng.normal(250, 150, n)
    tenure_months = (seg == 3)*rng.normal(6, 4, n) + \
        (seg != 3)*rng.normal(36, 20, n)
    returns = rng.poisson(lam=(seg == 0)*0.6 + (seg == 2)
                          * 0.8 + (seg == 1)*0.2 + (seg == 3)*0.3, size=n)

    df = pd.DataFrame({
        "CustomerID": range(1, n+1),
        "Income": np.clip(income, 15000, 200000),
        "RecencyDays": np.clip(recency_days, 0, None),
        "Frequency": np.clip(frequency, 0, None),
        "Monetary": np.clip(monetary, 0, None),
        "TenureMonths": np.clip(tenure_months, 0, None),
        "Returns": np.clip(returns, 0, None),
    })
    return df


if demo:
    df = load_demo_data()
else:
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or tick 'Use demo synthetic data' to proceed.")
        st.stop()

st.subheader("📄 Data Preview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Rows: **{df.shape[0]}**  |  Columns: **{df.shape[1]}**")

# ---------------------- FEATURE SELECTION ----------------------
st.header("2) Features & Preprocessing")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "CustomerID" in numeric_cols:
    numeric_cols.remove("CustomerID")

if not numeric_cols:
    st.error(
        "No numeric columns found. Please upload a dataset with numeric features.")
    st.stop()

feat_sel = st.multiselect("Select numeric features for clustering", options=numeric_cols,
                          default=[c for c in numeric_cols if c.lower() in {"recencydays", "frequency", "monetary", "income", "tenuremonths"}] or numeric_cols)

col_a, col_b, col_c = st.columns(3)
with col_a:
    scaler_name = st.selectbox(
        "Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
with col_b:
    impute_method = st.selectbox(
        "Missing values", ["Drop rows", "Fill with Median", "Fill with Mean"])
with col_c:
    standardize_outliers = st.checkbox(
        "Clip extreme outliers (1st–99th pct)", value=True)

if len(feat_sel) < 2:
    st.warning("Select at least two features.")
    st.stop()

X = df[feat_sel].copy()

# Missing values
if impute_method == "Drop rows":
    keep_mask = ~X.isna().any(axis=1)
    X = X[keep_mask]
    df_work = df.loc[X.index].copy()
else:
    df_work = df.copy()
    if impute_method == "Fill with Median":
        X = X.fillna(X.median(numeric_only=True))
    else:
        X = X.fillna(X.mean(numeric_only=True))

# Outlier clipping
if standardize_outliers:
    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1)

# Scaling
scaler = {"StandardScaler": StandardScaler(),
          "MinMaxScaler": MinMaxScaler(),
          "RobustScaler": RobustScaler()}[scaler_name]
X_scaled = scaler.fit_transform(X)

# ---------------------- DIMENSIONALITY REDUCTION ----------------------
st.header("3) Dimensionality Reduction (Optional)")
use_pca = st.checkbox("Apply PCA for visualization / speed", value=True)
pca_components = st.slider(
    "PCA components (2–5)", min_value=2, max_value=5, value=3) if use_pca else None

if use_pca:
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_.cumsum()
    st.write(f"Explained variance (cumulative): {np.round(exp_var, 3)}")
else:
    X_pca = None

# ---------------------- ALGORITHM & K SELECTION ----------------------
st.header("4) Clustering")
algo = st.selectbox(
    "Algorithm", ["KMeans", "GaussianMixture (GMM)", "Agglomerative"])
k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=4)

col1, col2 = st.columns(2)
with col1:
    auto_k = st.checkbox("Suggest k (Elbow + Silhouette)", value=False)
with col2:
    eval_space = st.selectbox(
        "Evaluate on", ["Scaled features", "PCA features (if enabled)"])

if auto_k:
    eval_X = X_pca if (eval_space.startswith(
        "PCA") and X_pca is not None) else X_scaled
    inertias = []
    sils = []
    ks = list(range(2, min(13, max(3, len(eval_X) // 10))))
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init="auto", random_state=42).fit(eval_X)
        inertias.append(km.inertia_)
        labels = km.labels_
        try:
            sil = silhouette_score(eval_X, labels)
        except Exception:
            sil = np.nan
        sils.append(sil)

    fig_elbow = px.line(x=ks, y=inertias, markers=True,
                        labels={"x": "k", "y": "Inertia"},
                        title="Elbow Curve (KMeans)")
    fig_sil = px.line(x=ks, y=sils, markers=True,
                      labels={"x": "k", "y": "Silhouette Score"},
                      title="Silhouette Scores (KMeans)")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.plotly_chart(fig_sil, use_container_width=True)
    if np.isfinite(sils).any():
        k_suggest = ks[int(np.nanargmax(sils))]
        st.info(f"Suggested k based on max silhouette: **{k_suggest}**")

# ---------------------- FIT MODEL ----------------------
run = st.button("🚀 Run Clustering")
if not run:
    st.stop()

use_X_for_fit = X_pca if (eval_space.startswith(
    "PCA") and X_pca is not None) else X_scaled

if algo == "KMeans":
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(use_X_for_fit)
    center_space = "PCA" if use_X_for_fit is X_pca else "Scaled"
    centers = model.cluster_centers_
elif algo == "GaussianMixture (GMM)":
    model = GaussianMixture(
        n_components=k, covariance_type="full", random_state=42)
    labels = model.fit_predict(use_X_for_fit)
    centers = model.means_
    center_space = "PCA" if use_X_for_fit is X_pca else "Scaled"
else:  # Agglomerative
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(use_X_for_fit)
    centers = None
    center_space = None

df_out = df_work.copy()
df_out["Cluster"] = labels

# ---------------------- EVALUATION ----------------------
try:
    sil = silhouette_score(use_X_for_fit, labels)
    st.success(
        f"Silhouette score: **{sil:.3f}** (higher is better, ~0.2–0.5 is common for customer data)")
except Exception:
    st.warning(
        "Silhouette score could not be computed (e.g., singleton clusters).")

# ---------------------- VISUALIZATION ----------------------
st.header("5) Visualizations")

# 2D plot (PCA 2 components)
if use_pca and X_pca.shape[1] >= 2:
    fig2d = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=labels.astype(str),
        labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
        title="2D PCA Scatter by Cluster",
        hover_data=[df_work.index]
    )
    st.plotly_chart(fig2d, use_container_width=True)

# 3D plot (PCA 3 components)
if use_pca and X_pca.shape[1] >= 3:
    fig3d = px.scatter_3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        color=labels.astype(str),
        labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Cluster"},
        title="3D PCA Scatter by Cluster"
    )
    st.plotly_chart(fig3d, use_container_width=True)

# Dendrogram (Agglomerative)
if algo == "Agglomerative":
    st.subheader("Dendrogram (sample up to 500 rows)")
    sample_idx = np.random.choice(len(use_X_for_fit), size=min(
        500, len(use_X_for_fit)), replace=False)
    Z = linkage(use_X_for_fit[sample_idx], method="ward")
    # Convert to simple coordinates for Plotly
    # For brevity, display using scipy dendrogram on a static image alternative:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns  # only used inside Streamlit runtime if available
    import tempfile
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title("Hierarchical Clustering Dendrogram (sampled)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", dpi=150)
    st.image(tmp.name, use_container_width=True)

# ---------------------- CLUSTER PROFILES ----------------------
st.header("6) Cluster Profiling")

# Aggregate stats
prof = df_out.groupby("Cluster")[feat_sel].agg(
    ["mean", "median", "std", "min", "max", "count"])
prof.columns = ["_".join(col) for col in prof.columns]
st.dataframe(prof, use_container_width=True)

# Normalized radar/table-like view
st.subheader("Feature Means by Cluster (Scaled 0–1 per feature)")
norm = (df_out.groupby("Cluster")[feat_sel].mean(
) - df_out[feat_sel].min()) / (df_out[feat_sel].max() - df_out[feat_sel].min())
norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0)
st.dataframe(norm.style.background_gradient(axis=0), use_container_width=True)

# Centers in chosen space
if centers is not None:
    st.subheader(f"Cluster Centers ({center_space} space)")
    st.dataframe(pd.DataFrame(centers, columns=[
                 f"Dim{i+1}" for i in range(centers.shape[1])]), use_container_width=True)

# ---------------------- NAMING / BUSINESS LABELS ----------------------
st.header("7) Optional: Auto-Name Segments (rule-of-thumb)")


def heuristic_names(profile_df):
    names = {}
    # Heuristic on scaled means
    means = (profile_df - profile_df.min()) / \
        (profile_df.max() - profile_df.min())
    means = means.replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in means.index:
        m = means.loc[c]
        label = []
        # Examples; adjust as needed for your features
        if "Monetary" in m.index:
            if m["Monetary"] > 0.7 and ("Frequency" not in m or m["Frequency"] > 0.6):
                label.append("Premium/Loyal")
            elif m["Monetary"] < 0.3:
                label.append("Low-Value")
        if "RecencyDays" in m.index:
            if m["RecencyDays"] < 0.3:
                label.append("Recent")
            elif m["RecencyDays"] > 0.7:
                label.append("At-Risk")
        if not label:
            label.append("General")
        names[c] = " & ".join(label)
    return names


if st.button("Suggest names"):
    names = heuristic_names(df_out.groupby("Cluster")[feat_sel].mean())
    name_map = pd.Series(names, name="SuggestedName")
    st.dataframe(name_map)

# ---------------------- DOWNLOADS ----------------------
st.header("8) Export")
# Add optional suggested names to df_out
if "SuggestedName" in locals() and isinstance(names, dict):
    df_out["SegmentName"] = df_out["Cluster"].map(names)

csv_bytes = df_out.to_csv(index=False).encode()
b64 = base64.b64encode(csv_bytes).decode()
href = f'<a href="data:text/csv;base64,{b64}" download="segmented_customers.csv">⬇️ Download segmented_customers.csv</a>'
st.markdown(href, unsafe_allow_html=True)

st.caption("Tip: Save your session by downloading the segmented CSV. Rerun with different features to compare.")
