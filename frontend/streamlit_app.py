"""
AI Revenue Risk Intelligence - Streamlit Frontend
Clean, human-readable UI with no emojis and high-contrast text.
"""

import io
import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from typing import Optional

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_URL = os.getenv("API_URL", "https://ai-revenue-assistance-1.onrender.com")
st.write("Current API URL:", API_URL)

st.set_page_config(
    page_title="Revenue Risk Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Styling — high contrast, no dark backgrounds on cards
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 2rem;
    }

    /* Cards use white background with dark text — fixes invisible text issue */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2563eb;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        color: #1a1a2e;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .risk-critical { border-left-color: #dc2626 !important; }
    .risk-high     { border-left-color: #ea580c !important; }
    .risk-medium   { border-left-color: #d97706 !important; }
    .risk-low      { border-left-color: #16a34a !important; }

    /* Action cards — white bg, dark readable text */
    .action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .action-card .action-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.4rem;
    }

    .action-card .action-rationale {
        font-size: 0.85rem;
        color: #444;
        margin-bottom: 0.25rem;
    }

    .action-card .action-impact {
        font-size: 0.82rem;
        color: #2563eb;
        font-style: italic;
    }

    /* Priority labels */
    .priority-1 { color: #dc2626; font-weight: 600; }
    .priority-2 { color: #ea580c; font-weight: 600; }
    .priority-3 { color: #d97706; font-weight: 600; }
    .priority-4 { color: #16a34a; font-weight: 600; }

    /* Risk badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-critical { background: #fee2e2; color: #dc2626; }
    .badge-high     { background: #ffedd5; color: #ea580c; }
    .badge-medium   { background: #fef3c7; color: #b45309; }
    .badge-low      { background: #dcfce7; color: #16a34a; }

    /* Section divider */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Sidebar — blue background with white text */
    [data-testid="stSidebar"] {
        background: #1e40af !important;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: #ffffff !important;
    }

    /* Selected radio item highlight */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
        background: rgba(255,255,255,0.15);
        border-radius: 6px;
    }

    /* Success/error alert boxes in sidebar */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }

    [data-testid="stSidebar"] .stAlert p {
        color: #ffffff !important;
    }

    /* Muted caption text */
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p {
        color: rgba(255,255,255,0.65) !important;
    }

    /* API URL code block */
    [data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.12) !important;
        color: #bfdbfe !important;
    }

    /* Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Summary blockquote */
    .summary-box {
        background: #f0f4ff;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        color: #1e3a5f;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

RISK_COLORS = {
    "Critical": "#dc2626",
    "High":     "#ea580c",
    "Medium":   "#d97706",
    "Low":      "#16a34a",
}

PRIORITY_LABELS = ["1st Priority", "2nd Priority", "3rd Priority", "4th Priority"]
PRIORITY_CLASSES = ["priority-1", "priority-2", "priority-3", "priority-4"]


def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the API at {API_URL}. Please make sure the backend is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_get(endpoint: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def risk_badge(risk_band: str) -> str:
    cls = f"badge badge-{risk_band.lower()}"
    return f'<span class="{cls}">{risk_band}</span>'


def probability_gauge(prob: float, risk_band: str) -> go.Figure:
    color = RISK_COLORS.get(risk_band, "#64748b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability (%)", "font": {"size": 16, "family": "IBM Plex Sans"}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#dcfce7"},
                {"range": [40, 60], "color": "#fef9c3"},
                {"range": [60, 80], "color": "#ffedd5"},
                {"range": [80, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor="white",
        font={"family": "IBM Plex Sans"},
    )
    return fig


def shap_bar_chart(factors: list) -> go.Figure:
    names  = [f["feature"] for f in factors]
    impacts = [f["impact"] for f in factors]
    colors  = ["#dc2626" if v > 0 else "#16a34a" for v in impacts]

    fig = go.Figure(go.Bar(
        y=names,
        x=impacts,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in impacts],
        textposition="outside",
        textfont={"size": 11},
    ))
    fig.update_layout(
        title={"text": "Top Risk Factors (SHAP)", "font": {"size": 14}},
        xaxis_title="SHAP Impact on Prediction",
        yaxis={"autorange": "reversed"},
        height=300,
        margin=dict(t=50, b=20, l=10, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "IBM Plex Sans", "color": "#1a1a2e"},
        xaxis={"gridcolor": "#f1f5f9"},
    )
    return fig


def render_action_cards(strategy: list):
    for action in strategy:
        idx = min(action["priority"] - 1, 3)
        label = PRIORITY_LABELS[idx]
        p_class = PRIORITY_CLASSES[idx]
        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">
                <span class="{p_class}">{label}:</span> {action['action']}
            </div>
            <div class="action-rationale">{action['rationale']}</div>
            <div class="action-impact">Expected impact: {action['expected_impact']}</div>
        </div>""", unsafe_allow_html=True)


def build_customer_form(prefix: str = "") -> dict:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender     = st.selectbox("Gender", ["Male", "Female"], key=f"{prefix}gender")
        senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key=f"{prefix}senior")
        partner    = st.selectbox("Partner", ["Yes", "No"], key=f"{prefix}partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key=f"{prefix}dep")
        tenure     = st.slider("Tenure (months)", 0, 72, 12, key=f"{prefix}tenure")

    with col2:
        st.markdown("**Services**")
        phone_service     = st.selectbox("Phone Service", ["Yes", "No"], key=f"{prefix}phone")
        multiple_lines    = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key=f"{prefix}ml")
        internet          = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key=f"{prefix}internet")
        online_security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key=f"{prefix}os")
        online_backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key=f"{prefix}ob")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key=f"{prefix}dp")
        tech_support      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key=f"{prefix}ts")
        streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key=f"{prefix}stv")
        streaming_movies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key=f"{prefix}sm")

    with col3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key=f"{prefix}contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key=f"{prefix}paper")
        payment  = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key=f"{prefix}pay",
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5, key=f"{prefix}mc")
        total   = st.number_input(
            "Total Charges ($)", 0.0, 10000.0,
            float(monthly * max(tenure, 1)),
            step=1.0, key=f"{prefix}tc",
        )

    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


# ──────────────────────────────────────────────
# Page: Single Customer Analysis
# ──────────────────────────────────────────────

def page_single_customer():
    st.markdown('<div class="main-header">Single Customer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyse churn risk and revenue exposure for an individual customer</div>', unsafe_allow_html=True)

    customer_id = st.text_input("Customer ID (optional)", placeholder="e.g. CUST-001")

    with st.form("single_customer_form"):
        customer_data = build_customer_form("sc_")
        submitted = st.form_submit_button("Run Full Analysis", use_container_width=True)

    if submitted:
        with st.spinner("Running analysis, please wait..."):
            payload = {"customer_id": customer_id or None, "customer": customer_data}
            result  = api_post("/full-analysis", payload)

        if result is None:
            return

        pred      = result["prediction"]
        rev       = result["revenue_metrics"]
        expl      = result["explainability"]
        risk_band = result["risk_band"]
        strategy  = result["retention_strategy"]
        summary   = result["generated_summary"]

        st.markdown("---")

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Churn Probability",  f"{pred['churn_probability']:.1%}")
        k2.metric("Risk Band",          risk_band)
        k3.metric("Customer Lifetime Value", f"${rev['customer_lifetime_value']:,.0f}")
        k4.metric("Revenue at Risk",    f"${rev['revenue_at_risk']:,.0f}")
        k5.metric("Risk Score",         f"{rev['revenue_risk_score']:.0f} / 100")

        # Charts
        col_gauge, col_shap = st.columns(2)
        with col_gauge:
            st.plotly_chart(probability_gauge(pred["churn_probability"], risk_band), use_container_width=True)
        with col_shap:
            st.plotly_chart(shap_bar_chart(expl["top_risk_factors"]), use_container_width=True)

        # AI Summary
        st.markdown('<div class="section-title">AI Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        # Retention Strategy
        st.markdown('<div class="section-title">Retention Strategy</div>', unsafe_allow_html=True)
        render_action_cards(strategy)

        st.info(f"Inference time: {result['inference_time_ms']:.0f} ms")


# ──────────────────────────────────────────────
# Page: Batch CSV Upload
# ──────────────────────────────────────────────

def page_batch_upload():
    st.markdown('<div class="main-header">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file of customers for portfolio-level risk analysis</div>', unsafe_allow_html=True)

    st.info("The CSV must contain the same columns as the IBM Telco Churn dataset (without customerID and Churn columns).")

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded is None:
        st.markdown("""
        **Required columns:**
        `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges`
        """)
        return

    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} customers successfully.")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Analyse All Customers", use_container_width=True):
        for drop_col in ["customerID", "Churn", "Churn_Binary"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        customers = df.to_dict(orient="records")

        with st.spinner(f"Analysing {len(customers)} customers..."):
            result = api_post("/dashboard", {"customers": customers})

        if result is None:
            return

        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers",       result["total_customers"])
        k2.metric("Average Churn Risk",    f"{result['avg_churn_probability']:.1%}")
        k3.metric("Total Revenue at Risk", f"${result['total_revenue_at_risk']:,.0f}")
        k4.metric("Recovery Opportunity",  f"${result['revenue_recovery_opportunity']:,.0f}")

        rd  = result["risk_distribution"]
        fig = px.pie(
            names=list(rd.keys()),
            values=list(rd.values()),
            title="Risk Distribution",
            color=list(rd.keys()),
            color_discrete_map=RISK_COLORS,
        )
        fig.update_layout(font={"family": "IBM Plex Sans"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Top At-Risk Customers</div>', unsafe_allow_html=True)
        top_df = pd.DataFrame(result["top_risk_customers"])
        st.dataframe(top_df.style.background_gradient(subset=["revenue_risk_score"], cmap="Reds"), use_container_width=True)

        csv_bytes = top_df.to_csv(index=False).encode()
        st.download_button(
            "Download Top Risk Customers as CSV",
            data=csv_bytes,
            file_name="top_risk_customers.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# Page: Revenue Dashboard
# ──────────────────────────────────────────────

def page_revenue_dashboard():
    st.markdown('<div class="main-header">Revenue Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Portfolio-level revenue risk monitoring</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Customer CSV for Dashboard", type=["csv"])
    if uploaded is None:
        st.info("Upload a customer CSV file to generate the revenue dashboard.")
        return

    df = pd.read_csv(uploaded)
    for drop_col in ["customerID", "Churn", "Churn_Binary"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    customers = df.to_dict(orient="records")

    with st.spinner("Computing revenue metrics..."):
        result = api_post("/dashboard", {"customers": customers})

    if result is None:
        return

    rd     = result["risk_distribution"]
    top_df = pd.DataFrame(result["top_risk_customers"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",       f"{result['total_customers']:,}")
    k2.metric("Average Churn Risk",    f"{result['avg_churn_probability']:.1%}")
    k3.metric("Total Revenue at Risk", f"${result['total_revenue_at_risk']:,.0f}")
    k4.metric("Recovery Opportunity",  f"${result['revenue_recovery_opportunity']:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            names=list(rd.keys()),
            values=list(rd.values()),
            title="Risk Band Distribution",
            hole=0.4,
            color=list(rd.keys()),
            color_discrete_map=RISK_COLORS,
        )
        fig_pie.update_layout(font={"family": "IBM Plex Sans"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            top_df,
            x="customer_index",
            y="revenue_at_risk",
            color="risk_band",
            title="Top 10 Customers by Revenue at Risk",
            color_discrete_map=RISK_COLORS,
            labels={"customer_index": "Customer", "revenue_at_risk": "Revenue at Risk ($)"},
        )
        fig_bar.update_layout(font={"family": "IBM Plex Sans"}, plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    scatter_fig = px.scatter(
        top_df,
        x="churn_probability",
        y="clv",
        color="risk_band",
        size="revenue_at_risk",
        title="Customer Lifetime Value vs Churn Probability  (bubble size = Revenue at Risk)",
        color_discrete_map=RISK_COLORS,
        labels={"churn_probability": "Churn Probability", "clv": "CLV ($)"},
    )
    scatter_fig.update_layout(font={"family": "IBM Plex Sans"}, plot_bgcolor="white")
    st.plotly_chart(scatter_fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Explainability Dashboard
# ──────────────────────────────────────────────

def page_explainability():
    st.markdown('<div class="main-header">Explainability Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SHAP-based feature importance and risk factor analysis</div>', unsafe_allow_html=True)

    st.info("Enter a customer profile below to see which features are driving their churn risk.")

    with st.form("explain_form"):
        customer_data = build_customer_form("ex_")
        submitted = st.form_submit_button("Explain Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Computing SHAP values..."):
            pred_result = api_post("/predict", customer_data)
            shap_result = api_post("/explain", customer_data)

        if shap_result is None or pred_result is None:
            return

        prob    = pred_result["churn_probability"]
        factors = shap_result["top_risk_factors"]

        st.markdown("---")
        st.subheader(f"Churn Probability: {prob:.1%}")

        names   = [f["feature"] for f in factors]
        impacts = [f["impact"] for f in factors]
        colors  = ["#dc2626" if v > 0 else "#16a34a" for v in impacts]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names,
            x=impacts,
            orientation="h",
            marker_color=colors,
            text=[f"{'+' if v > 0 else ''}{v:.4f}" for v in impacts],
            textposition="outside",
        ))
        fig.update_layout(
            title="SHAP Feature Contributions — Red increases churn risk, Green decreases it",
            xaxis_title="SHAP Value (impact on prediction)",
            yaxis={"autorange": "reversed"},
            height=400,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "IBM Plex Sans", "color": "#1a1a2e"},
            xaxis={"gridcolor": "#f1f5f9"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Feature Impact Details</div>', unsafe_allow_html=True)
        factor_df = pd.DataFrame(factors)
        factor_df["direction"] = factor_df["impact"].apply(
            lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk"
        )
        st.dataframe(factor_df, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Retention Strategy Generator
# ──────────────────────────────────────────────

def page_retention():
    st.markdown('<div class="main-header">Retention Strategy Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered personalised retention playbook for at-risk customers</div>', unsafe_allow_html=True)

    customer_id = st.text_input("Customer ID", placeholder="CUST-001")

    with st.form("retention_form"):
        customer_data = build_customer_form("ret_")
        submitted = st.form_submit_button("Generate Retention Strategy", use_container_width=True)

    if submitted:
        with st.spinner("Generating retention strategy..."):
            payload = {"customer_id": customer_id or None, "customer": customer_data}
            result  = api_post("/recommend", payload)

        if result is None:
            return

        st.markdown("---")

        source = result.get("source", "")
        if source == "groq":
            st.success("Strategy generated by Groq AI (llama-3.3-70b-versatile)")
        else:
            st.info("Strategy generated by rule-based engine. Set GROQ_API_KEY for AI-powered strategies.")

        st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{result["generated_summary"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Why This Customer May Churn</div>', unsafe_allow_html=True)
        for reason in result.get("churn_reasons", []):
            st.markdown(f"- {reason}")

        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">{result.get("business_impact", "")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Recommended Action Plan</div>', unsafe_allow_html=True)
        render_action_cards(result["retention_strategy"])


# ──────────────────────────────────────────────
# Page: Model Monitoring
# ──────────────────────────────────────────────

def page_monitoring():
    st.markdown('<div class="main-header">Model Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System health and model performance tracking</div>', unsafe_allow_html=True)

    health = api_get("/health")
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status",     "Healthy"  if health["status"] == "healthy" else "Degraded")
        col2.metric("Model Loaded",   "Yes"      if health["model_loaded"]         else "No")
        col3.metric("Version",        health["version"])
    else:
        st.error("Cannot reach the API.")
        return

    st.markdown("---")
    st.markdown('<div class="section-title">Inference Benchmark</div>', unsafe_allow_html=True)

    sample_customer = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5, "TotalCharges": 427.5,
    }

    n_runs = st.slider("Number of inference runs", 1, 50, 10)

    if st.button("Run Benchmark"):
        times = []
        probs = []
        progress = st.progress(0)
        for i in range(n_runs):
            t0 = time.time()
            r  = api_post("/predict", sample_customer)
            elapsed = (time.time() - t0) * 1000
            if r:
                times.append(elapsed)
                probs.append(r["churn_probability"])
            progress.progress((i + 1) / n_runs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Latency (ms)", f"{sum(times)/len(times):.1f}")
        col2.metric("P95 Latency (ms)",     f"{sorted(times)[int(len(times)*0.95)]:.1f}")
        col3.metric("Total Runs",           n_runs)
        col4.metric("Average Probability",  f"{sum(probs)/len(probs):.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=times, mode="lines+markers", name="Latency (ms)",
                                  line={"color": "#2563eb"}))
        fig.update_layout(
            title="Inference Latency Per Run",
            xaxis_title="Run Number",
            yaxis_title="Latency (ms)",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "IBM Plex Sans", "color": "#1a1a2e"},
            xaxis={"gridcolor": "#f1f5f9"},
            yaxis={"gridcolor": "#f1f5f9"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("## Revenue Risk AI")
        st.markdown(f"**API:** `{API_URL}`")

        health = api_get("/health")
        if health and health.get("status") == "healthy":
            st.success("API Connected")
        else:
            st.error("API Offline")

        st.markdown("---")
        st.markdown("**Navigation**")

        pages = {
            "Single Customer Analysis": page_single_customer,
            "Batch CSV Upload":         page_batch_upload,
            "Revenue Dashboard":        page_revenue_dashboard,
            "Explainability":           page_explainability,
            "Retention Strategy":       page_retention,
            "Model Monitoring":         page_monitoring,
        }
        choice = st.radio("", list(pages.keys()), label_visibility="collapsed")

        st.markdown("---")
        st.caption("AI Revenue Risk Intelligence v1.0")
        st.caption("Powered by LightGBM + SHAP + Groq")

    pages[choice]()


if __name__ == "__main__":
    main()