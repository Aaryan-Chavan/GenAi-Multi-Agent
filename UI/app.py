import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analytics Agent",
    page_icon="🚀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ---- Status banner ---- */
    .status-banner {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
        border-radius: 10px;
        padding: 12px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #166534;
        font-weight: 500;
    }
    .status-dot {
        width: 10px; height: 10px;
        background: #22c55e;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ---- Section cards ---- */
    .card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px 24px;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---- Upload area ---- */
    .upload-box {
        border: 2px dashed #c4b5fd;
        border-radius: 12px;
        padding: 32px 16px;
        text-align: center;
        background: #faf5ff;
        color: #6d28d9;
        margin-bottom: 14px;
    }
    .upload-icon { font-size: 2rem; margin-bottom: 6px; }
    .upload-box p { margin: 4px 0; font-size: 0.85rem; color: #7c3aed; font-weight: 500; }
    .upload-box small { color: #9ca3af; font-size: 0.78rem; }

    /* ---- Uploaded file pill ---- */
    .file-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.85rem;
    }
    .file-icon { font-size: 1.4rem; }
    .file-name { font-weight: 600; color: #111827; }
    .file-meta { color: #6b7280; font-size: 0.78rem; }
    .file-badge {
        margin-left: auto;
        background: #dcfce7;
        color: #16a34a;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---- Dataset info rows ---- */
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.88rem;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #6b7280; display: flex; align-items: center; gap: 8px; }
    .info-value { font-weight: 600; color: #111827; }
    .info-value.success { color: #16a34a; }

    /* ---- Load in Agent banner ---- */
    .agent-banner {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .agent-left { display: flex; align-items: center; gap: 12px; }
    .agent-title { font-weight: 700; font-size: 1rem; color: #1e1b4b; }
    .agent-sub { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
    }
    .metric-label { font-size: 0.78rem; color: #6b7280; margin-bottom: 4px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
    .metric-value.purple { color: #7c3aed; }
    .metric-value.orange { color: #f97316; }
    .metric-sub { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; }
    .metric-icon { float: right; font-size: 1.4rem; opacity: 0.6; }

    /* ---- AI response ---- */
    .ai-response {
        background: #f9fafb;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #374151;
        margin-bottom: 1rem;
        border-left: 3px solid #7c3aed;
    }
    .ai-label { font-weight: 700; color: #7c3aed; margin-bottom: 6px; font-size: 0.85rem; }

    /* ---- Revenue table ---- */
    .rev-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .rev-table th {
        background: #f9fafb;
        color: #6b7280;
        font-weight: 600;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }
    .rev-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f3f4f6;
        color: #374151;
    }
    .rev-table tr:last-child td {
        font-weight: 700;
        color: #111827;
        border-bottom: none;
    }

    /* ---- Example query chips ---- */
    .chip {
        display: inline-block;
        background: #f3f4f6;
        border-radius: 20px;
        padding: 5px 13px;
        font-size: 0.78rem;
        color: #374151;
        margin: 3px 3px 3px 0;
        cursor: pointer;
        border: 1px solid #e5e7eb;
    }

    /* ---- Streamlit overrides ---- */
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
        border: 1.5px solid #e5e7eb !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stFileUploader"] {
        border: none !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        font-size: 0.9rem !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    h1, h2, h3 { font-family: 'Inter', sans-serif !important; }

    /* hide default Streamlit header */
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False
if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False
if "query_result" not in st.session_state:
    st.session_state.query_result = False
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# ══════════════════════════════════════════════════════════════════════════════
# STATUS BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="status-banner">
    <div class="status-dot"></div>
    <strong>Status: 🔄 Agent Ready</strong>
    <span style="color:#166534;font-weight:400;margin-left:8px;">
        All systems are operational. You can now ask your questions.
    </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Upload + Dataset Info
# ══════════════════════════════════════════════════════════════════════════════
col_upload, col_info = st.columns([1, 1], gap="medium")

with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">☁️ 1. Upload Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload CSV / Excel / JSON / Parquet",
        type=["csv", "xlsx", "json", "parquet"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-box">
            <div class="upload-icon">☁️</div>
            <p><strong>Drag &amp; drop your file here</strong></p>
            <p>or click to browse</p>
            <small>Supports: CSV, Excel, JSON, Parquet (Max 5GB)</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        size_mb = uploaded_file.size / (1024 * 1024)
        size_str = f"{size_mb/1024:.1f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"

        # Try to get row count
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.DataFrame()
            rows = f"{len(df):,}"
            cols = len(df.columns)
        except Exception:
            rows = "N/A"
            cols = 0

        st.session_state.file_loaded = True
        st.session_state.rows = rows
        st.session_state.cols = cols
        st.session_state.size_str = size_str
        st.session_state.file_name = uploaded_file.name

        st.markdown(f"""
        <div class="upload-box" style="border-color:#86efac;background:#f0fdf4;">
            <div class="upload-icon">✅</div>
            <p style="color:#166534;"><strong>File uploaded successfully!</strong></p>
            <small style="color:#166534;">{uploaded_file.name}</small>
        </div>
        <div class="file-pill" style="margin-top:10px;">
            <span class="file-icon">📄</span>
            <div>
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-meta">{size_str} &bull; {rows} rows &bull; {uploaded_file.name.split('.')[-1].upper()}</div>
            </div>
            <div class="file-badge">✓ Uploaded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Dataset Information</div>', unsafe_allow_html=True)

    if st.session_state.file_loaded:
        fn = st.session_state.get("file_name", "customer_data_2024.csv")
        sz = st.session_state.get("size_str", "1.2 GB")
        rw = st.session_state.get("rows", "45,231")
        cl = st.session_state.get("cols", 24)
        structured = max(0, cl - 6)
    else:
        fn, sz, rw, cl, structured = "customer_data_2024.csv", "1.2 GB", "45,231", 24, 18

    upload_status = (
        '<span class="info-value success">✅ Uploaded Successfully</span>'
        if st.session_state.file_loaded
        else '<span style="color:#9ca3af;">—</span>'
    )

    st.markdown(f"""
    <div class="info-row">
        <span class="info-label">📄 File Name</span>
        <span class="info-value">{fn}</span>
    </div>
    <div class="info-row">
        <span class="info-label">📦 File Size</span>
        <span class="info-value">{sz}</span>
    </div>
    <div class="info-row">
        <span class="info-label">🔢 Total Rows</span>
        <span class="info-value">{rw}</span>
    </div>
    <div class="info-row">
        <span class="info-label">📐 Total Columns</span>
        <span class="info-value">{cl}</span>
    </div>
    <div class="info-row">
        <span class="info-label">📋 Structured Columns</span>
        <span class="info-value">{structured}</span>
    </div>
    <div class="info-row">
        <span class="info-label">🔀 Unstructured Columns</span>
        <span class="info-value">6</span>
    </div>
    <div class="info-row">
        <span class="info-label">🔄 Upload Status</span>
        {upload_status}
    </div>
    <div class="info-row">
        <span class="info-label">📅 Uploaded On</span>
        <span class="info-value">May 26, 2025 10:28 AM</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD IN AGENT BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
ban_col, btn_col = st.columns([3, 1], gap="medium")

with ban_col:
    st.markdown("""
    <div style="padding:16px 0 4px 4px;">
        <div style="font-size:1rem;font-weight:700;color:#1e1b4b;">🚀 Load in Agent</div>
        <div style="font-size:0.83rem;color:#6b7280;margin-top:3px;">
            Prepare your data and activate the multi-agent workflow
        </div>
    </div>
    """, unsafe_allow_html=True)

with btn_col:
    st.markdown("<div style='padding-top:12px'>", unsafe_allow_html=True)
    if st.button("🚀  Load in Agent", use_container_width=True):
        with st.spinner("Activating agent…"):
            time.sleep(1.2)
        st.session_state.agent_loaded = True
        st.success("Agent activated! You can now ask questions.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1.5px solid #f3f4f6;margin:6px 0 14px 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ask Your Question
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💬 2. Ask Your Question</div>', unsafe_allow_html=True)

query = st.text_area(
    label="Query input",
    placeholder="Type your question about the data…",
    height=110,
    max_chars=2000,
    label_visibility="collapsed",
    key="query_input",
)
char_count = len(query)
st.markdown(
    f'<div style="text-align:right;font-size:0.78rem;color:#9ca3af;margin-top:-10px;">'
    f'{char_count} / 2000</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div style="font-size:0.82rem;color:#6b7280;margin-bottom:8px;">
    <strong>Example queries:</strong>
</div>
<div>
    <span class="chip">Show total revenue by product category</span>
    <span class="chip">Top 5 customers by purchase</span>
    <span class="chip">Monthly sales trend for last year</span>
    <span class="chip">Average order value by region</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
_, submit_col = st.columns([4, 1])
with submit_col:
    submit = st.button("✈️  Submit Query", use_container_width=True)

if submit and query.strip():
    with st.spinner("Analyzing data…"):
        time.sleep(1.5)
    st.session_state.query_result = True
    st.session_state.query_text = query

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Query Result
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

res_col, dl_col = st.columns([4, 1])
with res_col:
    st.markdown('<div class="section-title">📊 3. Query Result</div>', unsafe_allow_html=True)
with dl_col:
    if st.session_state.query_result:
        st.download_button(
            label="⬇️ Download Result",
            data="Category,Total Revenue,% Contribution\nElectronics,$1050000,42.9\nClothing,$620000,25.3\n"
                 "Home Appliances,$450000,18.4\nBeauty & Health,$180000,7.3\nOthers,$150000,6.1",
            file_name="query_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

if st.session_state.query_result:
    # AI Response text
    st.markdown("""
    <div class="ai-response">
        <div class="ai-label">✨ AI Response</div>
        The total revenue by product category is shown below. Electronics has the highest revenue contribution,
        followed by Clothing and Home Appliances.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4, gap="small")
    kpis = [
        ("Total Revenue", "$2.45M", "Overall Revenue", "💰", ""),
        ("Top Category", "Electronics", "Highest Revenue", "🏆", "purple"),
        ("Total Orders", "12,456", "Total Orders Count", "🛒", ""),
        ("Avg. Order Value", "$196.67", "Per Order Average", "📈", "orange"),
    ]
    for col, (label, value, sub, icon, color_cls) in zip([m1, m2, m3, m4], kpis):
        with col:
            cls = f"metric-value {color_cls}".strip()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label} <span class="metric-icon">{icon}</span></div>
                <div class="{cls}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Chart + Table ─────────────────────────────────────────────────────────
    chart_col, table_col = st.columns([1, 1], gap="medium")

    revenue_data = {
        "Category":      ["Electronics", "Clothing", "Home Appliances", "Beauty & Health", "Others"],
        "Revenue":       [1_050_000,       620_000,    450_000,            180_000,           150_000],
        "Contribution":  [42.9,            25.3,       18.4,               7.3,               6.1],
        "Colors":        ["#7c3aed",       "#22c55e",  "#f59e0b",          "#f97316",         "#60a5fa"],
    }

    with chart_col:
        st.markdown('<div class="section-title" style="font-size:0.92rem;">Revenue by Product Category</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=revenue_data["Category"],
            values=revenue_data["Revenue"],
            hole=0.55,
            marker=dict(colors=revenue_data["Colors"]),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=0.72, y=0.5, font=dict(size=12)),
            margin=dict(l=0, r=0, t=10, b=10),
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="<b>$2.45M</b><br>Total",
                x=0.35, y=0.5,
                font_size=14,
                showarrow=False,
            )],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with table_col:
        st.markdown('<div class="section-title" style="font-size:0.92rem;">Revenue by Product Category</div>',
                    unsafe_allow_html=True)
        rows_html = ""
        for i, (cat, rev, pct) in enumerate(
            zip(revenue_data["Category"], revenue_data["Revenue"], revenue_data["Contribution"]), 1
        ):
            rows_html += f"""
            <tr>
                <td>{i}</td>
                <td>{cat}</td>
                <td>${rev:,}</td>
                <td>{pct}%</td>
            </tr>"""

        st.markdown(f"""
        <table class="rev-table">
            <thead>
                <tr>
                    <th>#</th><th>Category</th><th>Total Revenue</th><th>% Contribution</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
                <tr>
                    <td colspan="2">Total</td>
                    <td>$2,450,000</td>
                    <td>100%</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#9ca3af;">
        <div style="font-size:2.5rem;margin-bottom:8px;">📊</div>
        <div style="font-weight:600;color:#6b7280;font-size:0.95rem;">No results yet</div>
        <div style="font-size:0.83rem;margin-top:4px;">Upload a dataset and submit a query to see results here.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)