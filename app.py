"""
app.py -- web interface built with Streamlit.
Run: streamlit run app.py
"""

import os
import json
import numpy as np
import joblib
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config -- must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedicalAI - Unsupervised Learning",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = "saved_model"
DATA_PATH = "data/raw"

# ---------------------------------------------------------------------------
# Global CSS: dark navy background, liquid glass cards, aurora gradient,
# Apple-style typography (SF Pro / system-ui)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & base ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, 'SF Pro Display', 'Inter', sans-serif;
    background: #020b1a !important;
    color: #e8edf5 !important;
}

.stApp {
    background: #020b1a !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 1100px !important;
}

/* ── Aurora background ───────────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%,  rgba(0,210,180,0.13) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 5%,   rgba(80,120,255,0.16) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 50% 0%,   rgba(160,60,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 10% 80%,  rgba(0,180,255,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 90% 70%,  rgba(0,230,150,0.07) 0%, transparent 55%);
    animation: aurora 14s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes aurora {
    0%   { opacity: 0.7; transform: scale(1)   translateY(0px); }
    33%  { opacity: 1.0; transform: scale(1.04) translateY(-18px); }
    66%  { opacity: 0.85; transform: scale(0.97) translateY(10px); }
    100% { opacity: 0.9; transform: scale(1.02) translateY(-8px); }
}

/* ── Liquid glass card ───────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.6rem;
    position: relative;
    z-index: 1;
    box-shadow:
        0 0 0 0.5px rgba(255,255,255,0.06) inset,
        0 20px 60px rgba(0,0,0,0.35);
}

.glass-card-sm {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    z-index: 1;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

/* ── Hero header ─────────────────────────────────────────────── */
.hero-title {
    font-size: 3.6rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a8d4ff 45%, #00e8b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}

.hero-sub {
    font-size: 1.15rem;
    font-weight: 300;
    color: rgba(200,220,255,0.65);
    margin: 0 0 2.5rem;
    letter-spacing: 0.01em;
}

/* ── Section label ───────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(100,180,255,0.7);
    margin-bottom: 0.6rem;
}

/* ── Metric pill ─────────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.4rem;
}
.metric-pill {
    flex: 1;
    background: rgba(0,180,255,0.08);
    border: 1px solid rgba(0,180,255,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-pill .val {
    font-size: 1.7rem;
    font-weight: 500;
    color: #fff;
    display: block;
}
.metric-pill .lbl {
    font-size: 0.75rem;
    color: rgba(180,210,255,0.55);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    display: block;
    margin-top: 3px;
}

/* ── Result badge ────────────────────────────────────────────── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    border-radius: 50px;
    padding: 0.8rem 1.6rem;
    font-size: 1.25rem;
    font-weight: 500;
    margin-top: 1.2rem;
    letter-spacing: -0.01em;
}
.result-normal {
    background: rgba(0,230,120,0.12);
    border: 1px solid rgba(0,230,120,0.30);
    color: #00e878;
}
.result-pneumonia {
    background: rgba(255,80,80,0.12);
    border: 1px solid rgba(255,80,80,0.28);
    color: #ff6060;
}
.result-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.dot-normal     { background: #00e878; box-shadow: 0 0 8px #00e878; }
.dot-pneumonia  { background: #ff6060; box-shadow: 0 0 8px #ff6060; }

/* ── Confidence bar ──────────────────────────────────────────── */
.conf-track {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}
.conf-fill-normal    { background: linear-gradient(90deg, #00c46e, #00ffaa); }
.conf-fill-pneumonia { background: linear-gradient(90deg, #e03030, #ff7070); }
.conf-label {
    font-size: 0.78rem;
    color: rgba(180,210,255,0.5);
    margin-top: 6px;
    display: block;
}

/* ── Sample grid ─────────────────────────────────────────────── */
.sample-img-wrap {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
}
.sample-tag {
    position: absolute;
    bottom: 6px; left: 6px;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 6px;
}
.tag-normal    { background: rgba(0,200,100,0.7); color: #fff; }
.tag-pneumonia { background: rgba(220,50,50,0.7); color: #fff; }

/* ── Upload zone ─────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(100,180,255,0.25) !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(100,180,255,0.5) !important;
}

/* ── Misc ────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }
img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_extractor():
    from keras.applications.resnet50 import ResNet50
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(MODEL_DIR, "pipeline.pkl"))


@st.cache_data
def load_viz_data():
    features_2d     = np.load(os.path.join(MODEL_DIR, "features_2d.npy"))
    cluster_labels  = np.load(os.path.join(MODEL_DIR, "cluster_labels.npy"))
    original_labels = np.load(os.path.join(MODEL_DIR, "original_labels.npy"), allow_pickle=True)
    return features_2d, cluster_labels, original_labels


def build_cluster_map(cluster_labels, original_labels):
    cluster_map = {}
    for cid in np.unique(cluster_labels):
        mask   = cluster_labels == cid
        subset = original_labels[mask]
        unique, counts = np.unique(subset, return_counts=True)
        cluster_map[int(cid)] = unique[np.argmax(counts)]
    return cluster_map


def predict(pil_img, extractor, pipeline):
    """Returns (cluster_id, class_name, confidence_pct)."""
    from keras.applications.resnet50 import preprocess_input
    img  = pil_img.resize((224, 224)).convert("RGB")
    x    = np.array(img, dtype=np.float32)
    x    = np.expand_dims(x, axis=0)
    x    = preprocess_input(x)
    feat = extractor.predict(x, verbose=0).flatten().reshape(1, -1)
    cid  = int(pipeline.predict(feat)[0])
    feat_pca  = pipeline["pca"].transform(feat)
    distances = pipeline["kmeans"].transform(feat_pca)[0]
    conf = (1 - distances[cid] / distances.sum()) * 100
    return cid, conf


# ---------------------------------------------------------------------------
# Guard: model must exist
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(MODEL_DIR, "pipeline.pkl")):
    st.error("Model not found. Run train.py first.")
    st.stop()

with st.spinner("Loading models..."):
    extractor                              = load_extractor()
    pipeline                               = load_pipeline()
    features_2d, cluster_labels, orig_lbl = load_viz_data()
    cluster_map                            = build_cluster_map(cluster_labels, orig_lbl)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding: 4rem 0 2rem; position: relative; z-index: 1;">
    <p class="section-label">AI-Powered Analysis</p>
    <h1 class="hero-title">Lung Scan<br>Analyzer</h1>
    <p class="hero-sub">
        Upload a chest X-ray and get an instant AI assessment —
        no labels, no supervision, just pattern recognition.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scatter plot -- neon points on dark navy, Plotly (no matplotlib)
# We build an interactive Plotly chart styled to match the design
# ---------------------------------------------------------------------------
import plotly.graph_objects as go

labels_text = ["Healthy" if l == "NORMAL" else "Pneumonia" for l in orig_lbl]
colors      = ["#00e8b0" if l == "NORMAL" else "#ff5c8a" for l in orig_lbl]

fig = go.Figure()

# Healthy cluster
mask_n = np.array(orig_lbl) == "NORMAL"
fig.add_trace(go.Scatter(
    x=features_2d[mask_n, 0],
    y=features_2d[mask_n, 1],
    mode='markers',
    name='Healthy',
    marker=dict(
        color='rgba(0,232,176,0.75)',
        size=6,
        line=dict(color='rgba(0,232,176,0.0)', width=0),
    ),
    hovertemplate='<b>Healthy</b><extra></extra>',
))

# Pneumonia cluster
mask_p = np.array(orig_lbl) == "PNEUMONIA"
fig.add_trace(go.Scatter(
    x=features_2d[mask_p, 0],
    y=features_2d[mask_p, 1],
    mode='markers',
    name='Pneumonia',
    marker=dict(
        color='rgba(255,92,138,0.75)',
        size=6,
        line=dict(color='rgba(255,92,138,0.0)', width=0),
    ),
    hovertemplate='<b>Pneumonia</b><extra></extra>',
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(6,18,42,0.75)',
    font=dict(family='-apple-system, SF Pro Display, Inter, sans-serif', color='#8ab4d8'),
    title=dict(
        text='How the AI grouped the scans',
        font=dict(size=16, color='#c8dff5'),
        x=0,
        pad=dict(l=0, b=12),
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.04)',
        bordercolor='rgba(255,255,255,0.08)',
        borderwidth=1,
        font=dict(size=12, color='#aac4e0'),
        itemsizing='constant',
        orientation='h',
        yanchor='bottom', y=1.02,
        xanchor='left', x=0,
    ),
    xaxis=dict(
        title=dict(text='Pattern dimension 1', font=dict(size=11, color='rgba(140,180,220,0.5)')),
        tickfont=dict(size=10),
        gridcolor='rgba(255,255,255,0.04)',
        zeroline=False,
        showline=False,
    ),
    yaxis=dict(
        title=dict(text='Pattern dimension 2', font=dict(size=11, color='rgba(140,180,220,0.5)')),
        tickfont=dict(size=10),
        gridcolor='rgba(255,255,255,0.04)',
        zeroline=False,
        showline=False,
    ),
    margin=dict(l=10, r=10, t=54, b=10),
    height=420,
)

st.markdown('<div class="glass-card" style="padding: 1.8rem 2rem;">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("""
<p style="font-size:0.82rem; color:rgba(140,180,220,0.45); margin:0.4rem 0 0;">
Each dot is a chest X-ray. The AI found these two groups on its own,
without ever being told what to look for.
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------
if os.path.exists(DATA_PATH):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Dataset samples</p>', unsafe_allow_html=True)

    sample_imgs, sample_labels = [], []
    for category in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(DATA_PATH, category)
        if os.path.isdir(folder):
            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:5]
            for f in files:
                sample_imgs.append(os.path.join(folder, f))
                sample_labels.append(category)

    cols = st.columns(5, gap="small")
    for i, (img_path, label) in enumerate(zip(sample_imgs[:10], sample_labels[:10])):
        tag_cls  = "tag-normal" if label == "NORMAL" else "tag-pneumonia"
        tag_text = "Healthy" if label == "NORMAL" else "Pneumonia"
        with cols[i % 5]:
            st.markdown(f'<div class="sample-img-wrap">', unsafe_allow_html=True)
            st.image(Image.open(img_path).resize((220, 220)), use_container_width=True)
            st.markdown(
                f'<span class="sample-tag {tag_cls}">{tag_text}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predict section
# ---------------------------------------------------------------------------
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card" style="background: rgba(60,120,255,0.055); border-color: rgba(100,160,255,0.14);">
<p class="section-label">Analyze your scan</p>
<p style="font-size:1.05rem; color:rgba(200,220,255,0.7); margin:0 0 1.2rem; font-weight:300;">
    Drop a chest X-ray below and the model will tell you what it sees.
</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    pil_img = Image.open(uploaded)

    # Run prediction before splitting into columns
    with st.spinner("Analyzing..."):
        cid, conf = predict(pil_img, extractor, pipeline)

    class_name  = cluster_map.get(cid, f"Cluster {cid}")
    is_normal   = class_name == "NORMAL"
    disp_name   = "Healthy" if is_normal else "Pneumonia"
    conf_r      = round(conf, 1)

    # Build color values based on result
    if is_normal:
        badge_bg    = "rgba(0,230,120,0.12)"
        badge_bdr   = "rgba(0,230,120,0.30)"
        badge_color = "#00e878"
        dot_color   = "#00e878"
        dot_shadow  = "#00e878"
        bar_grad    = "linear-gradient(90deg, #00c46e, #00ffaa)"
    else:
        badge_bg    = "rgba(255,80,80,0.12)"
        badge_bdr   = "rgba(255,80,80,0.28)"
        badge_color = "#ff6060"
        dot_color   = "#ff6060"
        dot_shadow  = "#ff6060"
        bar_grad    = "linear-gradient(90deg, #e03030, #ff7070)"

    # Build the entire result HTML as a plain string (no f-string nesting issues)
    result_html = (
        '<div style="margin-top:1.2rem;">'

        # Result label
        '<p style="font-size:0.72rem; color:rgba(160,200,255,0.5); '
        'letter-spacing:0.1em; text-transform:uppercase; margin:0 0 0.6rem;">Result</p>'

        # Badge
        '<div style="display:inline-flex; align-items:center; gap:10px; '
        'border-radius:50px; padding:0.75rem 1.5rem; font-size:1.2rem; font-weight:500; '
        'background:' + badge_bg + '; border:1px solid ' + badge_bdr + '; color:' + badge_color + ';">'
        '<span style="width:10px; height:10px; border-radius:50%; display:inline-block; '
        'background:' + dot_color + '; box-shadow:0 0 8px ' + dot_shadow + ';"></span>'
        + disp_name +
        '</div>'

        # Confidence label
        '<div style="margin-top:1.6rem;">'
        '<p style="font-size:0.72rem; color:rgba(160,200,255,0.5); '
        'letter-spacing:0.1em; text-transform:uppercase; margin:0 0 0.3rem;">Confidence</p>'

        # Confidence number
        '<p style="font-size:2rem; font-weight:500; margin:0; color:#fff;">'
        + str(conf_r) + '%'
        '</p>'

        # Progress bar
        '<div style="width:100%; height:6px; background:rgba(255,255,255,0.08); '
        'border-radius:99px; margin-top:0.8rem; overflow:hidden;">'
        '<div style="height:100%; border-radius:99px; width:' + str(conf_r) + '%; '
        'background:' + bar_grad + ';"></div>'
        '</div>'
        '<span style="font-size:0.75rem; color:rgba(180,210,255,0.4); margin-top:6px; display:block;">'
        'How strongly the scan aligns with this group</span>'
        '</div>'

        # Metric pills
        '<div style="display:flex; gap:1rem; margin-top:1.4rem;">'

        '<div style="flex:1; background:rgba(0,180,255,0.08); border:1px solid rgba(0,180,255,0.18); '
        'border-radius:14px; padding:1rem 1.2rem; text-align:center;">'
        '<span style="font-size:1.6rem; font-weight:500; color:#fff; display:block;">'
        + str(cid) +
        '</span>'
        '<span style="font-size:0.72rem; color:rgba(180,210,255,0.5); letter-spacing:0.06em; '
        'text-transform:uppercase; display:block; margin-top:3px;">Cluster</span>'
        '</div>'

        '<div style="flex:1; background:rgba(0,180,255,0.08); border:1px solid rgba(0,180,255,0.18); '
        'border-radius:14px; padding:1rem 1.2rem; text-align:center;">'
        '<span style="font-size:1.6rem; font-weight:500; color:#fff; display:block;">'
        + disp_name +
        '</span>'
        '<span style="font-size:0.72rem; color:rgba(180,210,255,0.5); letter-spacing:0.06em; '
        'text-transform:uppercase; display:block; margin-top:3px;">Group</span>'
        '</div>'

        '</div>'

        # Disclaimer
        '<p style="font-size:0.7rem; color:rgba(150,180,220,0.3); margin-top:1.4rem; line-height:1.6;">'
        'This is a research model, not a medical diagnostic tool. '
        'Always consult a qualified physician.</p>'

        '</div>'
    )

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(pil_img, use_container_width=True)

    with col_res:
        st.markdown(result_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:3rem 0 1rem; position:relative; z-index:1;">
    <p style="font-size:0.75rem; color:rgba(140,170,210,0.3); letter-spacing:0.06em;">
        LUNG SCAN ANALYZER &nbsp;&bull;&nbsp; UNSUPERVISED LEARNING PROJECT
    </p>
</div>
""", unsafe_allow_html=True)