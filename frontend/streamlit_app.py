"""
AI Revenue Risk Intelligence - Streamlit Frontend
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

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Revenue Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .risk-critical { border-left-color: #dc3545 !important; }
    .risk-high     { border-left-color: #fd7e14 !important; }
    .risk-medium   { border-left-color: #ffc107 !important; }
    .risk-low      { border-left-color: #28a745 !important; }
    .action-card {
        background: #fff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .badge-critical { background-color: #dc3545; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-high     { background-color: #fd7e14; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-medium   { background-color: #ffc107; color: black;  padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-low      { background-color: #28a745; color: white;  padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

RISK_COLORS = {
    "Critical": "#dc3545",
    "High": "#fd7e14",
    "Medium": "#ffc107",
    "Low": "#28a745",
}

def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_get(endpoint: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


def risk_badge(risk_band: str) -> str:
    cls = f"badge-{risk_band.lower()}"
    return f'<span class="{cls}">{risk_band}</span>'


def probability_gauge(prob: float, risk_band: str) -> go.Figure:
    color = RISK_COLORS.get(risk_band, "#6c757d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability %", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#d4edda"},
                {"range": [40, 60], "color": "#fff3cd"},
                {"range": [60, 80], "color": "#fde8d8"},
                {"range": [80, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def shap_bar_chart(factors: list) -> go.Figure:
    names = [f["feature"] for f in factors]
    impacts = [f["impact"] for f in factors]
    colors = ["#dc3545" if v > 0 else "#28a745" for v in impacts]

    fig = go.Figure(go.Bar(
        y=names,
        x=impacts,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in impacts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top Risk Factors (SHAP)",
        xaxis_title="SHAP Impact",
        yaxis={"autorange": "reversed"},
        height=320,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def build_customer_form(prefix: str = "") -> dict:
    """Render the customer input form and return a dict."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"{prefix}gender")
        senior = st.selectbox("Senior Citizen", [0, 1], key=f"{prefix}senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key=f"{prefix}partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key=f"{prefix}dep")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{prefix}tenure")

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key=f"{prefix}phone")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key=f"{prefix}ml")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key=f"{prefix}internet")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key=f"{prefix}os")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key=f"{prefix}ob")
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key=f"{prefix}dp")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key=f"{prefix}ts")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key=f"{prefix}stv")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key=f"{prefix}sm")

    with col3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key=f"{prefix}contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key=f"{prefix}paper")
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key=f"{prefix}pay",
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5, key=f"{prefix}mc")
        total = st.number_input(
            "Total Charges ($)",
            0.0, 10000.0,
            float(monthly * max(tenure, 1)),
            step=1.0,
            key=f"{prefix}tc",
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
    st.markdown('<div class="main-header">🎯 Single Customer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyse churn risk and revenue exposure for an individual customer</div>', unsafe_allow_html=True)

    customer_id = st.text_input("Customer ID (optional)", placeholder="e.g. CUST-001")

    with st.form("single_customer_form"):
        customer_data = build_customer_form("sc_")
        submitted = st.form_submit_button("🔍 Run Full Analysis", use_container_width=True)

    if submitted:
        with st.spinner("Running analysis..."):
            payload = {
                "customer_id": customer_id or None,
                "customer": customer_data,
            }
            result = api_post("/full-analysis", payload)

        if result is None:
            return

        pred = result["prediction"]
        rev = result["revenue_metrics"]
        expl = result["explainability"]
        risk_band = result["risk_band"]
        strategy = result["retention_strategy"]
        summary = result["generated_summary"]

        # ── Top KPIs ──
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Churn Probability", f"{pred['churn_probability']:.1%}")
        k2.metric("Risk Band", risk_band)
        k3.metric("CLV", f"${rev['customer_lifetime_value']:,.0f}")
        k4.metric("Revenue At Risk", f"${rev['revenue_at_risk']:,.0f}")
        k5.metric("Risk Score", f"{rev['revenue_risk_score']}/100")

        # ── Charts ──
        col_gauge, col_shap = st.columns(2)
        with col_gauge:
            st.plotly_chart(probability_gauge(pred["churn_probability"], risk_band), use_container_width=True)
        with col_shap:
            st.plotly_chart(shap_bar_chart(expl["top_risk_factors"]), use_container_width=True)

        # ── AI Summary ──
        st.markdown("---")
        st.subheader("🤖 AI Executive Summary")
        color = RISK_COLORS.get(risk_band, "#6c757d")
        st.markdown(f"""
        <div class="metric-card risk-{risk_band.lower()}">
        {summary}
        </div>""", unsafe_allow_html=True)

        # ── Retention Strategy ──
        st.subheader("🎯 Retention Strategy")
        for action in strategy:
            st.markdown(f"""
            <div class="action-card">
            <b>Priority {action['priority']}: {action['action']}</b><br>
            <small>📋 {action['rationale']}</small><br>
            <small>📈 Expected impact: <em>{action['expected_impact']}</em></small>
            </div>""", unsafe_allow_html=True)

        st.info(f"⚡ Inference time: {result['inference_time_ms']:.0f}ms")


# ──────────────────────────────────────────────
# Page: Batch CSV Upload
# ──────────────────────────────────────────────

def page_batch_upload():
    st.markdown('<div class="main-header">📁 Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV of customers for portfolio-level risk analysis</div>', unsafe_allow_html=True)

    st.info("CSV must contain the same columns as the IBM Telco Churn dataset (without customerID and Churn).")

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
    st.success(f"✅ Loaded {len(df)} customers")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("🚀 Analyse All Customers", use_container_width=True):
        # Drop irrelevant columns
        for drop_col in ["customerID", "Churn", "Churn_Binary"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        customers = df.to_dict(orient="records")

        with st.spinner(f"Analysing {len(customers)} customers..."):
            payload = {"customers": customers}
            result = api_post("/dashboard", payload)

        if result is None:
            return

        # Get per-customer predictions for the table
        preds_payload = {"customers": customers}
        preds_result = api_post("/dashboard", preds_payload)

        # KPIs
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", result["total_customers"])
        k2.metric("Avg Churn Risk", f"{result['avg_churn_probability']:.1%}")
        k3.metric("Total Revenue At Risk", f"${result['total_revenue_at_risk']:,.0f}")
        k4.metric("Recovery Opportunity", f"${result['revenue_recovery_opportunity']:,.0f}")

        # Risk distribution
        rd = result["risk_distribution"]
        fig = px.pie(
            names=list(rd.keys()),
            values=list(rd.values()),
            title="Risk Distribution",
            color=list(rd.keys()),
            color_discrete_map=RISK_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top risk customers table
        st.subheader("🔴 Top Risk Customers")
        top_df = pd.DataFrame(result["top_risk_customers"])
        st.dataframe(top_df.style.background_gradient(subset=["revenue_risk_score"], cmap="Reds"), use_container_width=True)

        # Download
        csv_bytes = top_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Top Risk Customers CSV",
            data=csv_bytes,
            file_name="top_risk_customers.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# Page: Revenue Dashboard
# ──────────────────────────────────────────────

def page_revenue_dashboard():
    st.markdown('<div class="main-header">💰 Revenue Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Portfolio-level revenue risk monitoring</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Customer CSV for Dashboard", type=["csv"])
    if uploaded is None:
        st.info("Upload a customer CSV to generate the revenue dashboard.")
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

    rd = result["risk_distribution"]
    top_df = pd.DataFrame(result["top_risk_customers"])

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Total Customers", f"{result['total_customers']:,}")
    k2.metric("⚠️ Avg Churn Risk", f"{result['avg_churn_probability']:.1%}")
    k3.metric("💸 Total Revenue At Risk", f"${result['total_revenue_at_risk']:,.0f}")
    k4.metric("💡 Recovery Opportunity", f"${result['revenue_recovery_opportunity']:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution donut
        fig_pie = px.pie(
            names=list(rd.keys()),
            values=list(rd.values()),
            title="Risk Band Distribution",
            hole=0.4,
            color=list(rd.keys()),
            color_discrete_map=RISK_COLORS,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Revenue at risk by risk band
        fig_bar = px.bar(
            top_df,
            x="customer_index",
            y="revenue_at_risk",
            color="risk_band",
            title="Top 10 Customers by Revenue At Risk",
            color_discrete_map=RISK_COLORS,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: CLV vs Churn Probability
    scatter_fig = px.scatter(
        top_df,
        x="churn_probability",
        y="clv",
        color="risk_band",
        size="revenue_at_risk",
        title="CLV vs Churn Probability (bubble = Revenue At Risk)",
        color_discrete_map=RISK_COLORS,
        labels={"churn_probability": "Churn Probability", "clv": "CLV ($)"},
    )
    st.plotly_chart(scatter_fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Explainability Dashboard
# ──────────────────────────────────────────────

def page_explainability():
    st.markdown('<div class="main-header">🔬 Explainability Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SHAP-based feature importance and risk factor analysis</div>', unsafe_allow_html=True)

    st.info("Enter a customer profile to see which features are driving their churn risk.")

    with st.form("explain_form"):
        customer_data = build_customer_form("ex_")
        submitted = st.form_submit_button("🔍 Explain Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Computing SHAP values..."):
            pred_result = api_post("/predict", customer_data)
            shap_result = api_post("/explain", customer_data)

        if shap_result is None:
            return

        prob = pred_result["churn_probability"]
        factors = shap_result["top_risk_factors"]

        st.markdown("---")
        st.subheader(f"Churn Probability: {prob:.1%}")

        # SHAP waterfall bar chart
        fig = go.Figure()
        names = [f["feature"] for f in factors]
        impacts = [f["impact"] for f in factors]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in impacts]

        fig.add_trace(go.Bar(
            y=names,
            x=impacts,
            orientation="h",
            marker_color=colors,
            text=[f"{'↑' if v > 0 else '↓'} {abs(v):.4f}" for v in impacts],
            textposition="outside",
        ))
        fig.update_layout(
            title="SHAP Feature Contributions (Red = Increases Churn Risk)",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis={"autorange": "reversed"},
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Feature Impact Details")
        factor_df = pd.DataFrame(factors)
        factor_df["direction"] = factor_df["impact"].apply(lambda x: "⬆️ Risk Increase" if x > 0 else "⬇️ Risk Decrease")
        st.dataframe(factor_df, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Retention Strategy Generator
# ──────────────────────────────────────────────

def page_retention():
    st.markdown('<div class="main-header">🎯 Retention Strategy Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered personalised retention playbook for at-risk customers</div>', unsafe_allow_html=True)

    customer_id = st.text_input("Customer ID", placeholder="CUST-001")

    with st.form("retention_form"):
        customer_data = build_customer_form("ret_")
        submitted = st.form_submit_button("🤖 Generate Retention Strategy", use_container_width=True)

    if submitted:
        with st.spinner("Generating AI retention strategy..."):
            payload = {"customer_id": customer_id or None, "customer": customer_data}
            result = api_post("/recommend", payload)

        if result is None:
            return

        st.markdown("---")

        source = result.get("source", "")
        if source == "groq":
            st.success("✅ Strategy generated by Groq AI (llama-3.3-70b-versatile)")
        else:
            st.info("ℹ️ Strategy generated by rule-based engine (set GROQ_API_KEY for AI strategies)")

        st.subheader("📋 Executive Summary")
        st.markdown(f"> {result['generated_summary']}")

        st.subheader("⚠️ Churn Reasons")
        for reason in result.get("churn_reasons", []):
            st.markdown(f"- {reason}")

        st.subheader("💸 Business Impact")
        st.markdown(result.get("business_impact", ""))

        st.subheader("🎯 Retention Action Plan")
        for action in result["retention_strategy"]:
            prio_color = ["🔴", "🟠", "🟡", "🟢"]
            icon = prio_color[min(action["priority"] - 1, 3)]
            st.markdown(f"""
            <div class="action-card">
            <b>{icon} Priority {action['priority']}: {action['action']}</b><br>
            <small>📋 {action['rationale']}</small><br>
            <small>📈 Expected impact: <em>{action['expected_impact']}</em></small>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Page: Model Monitoring
# ──────────────────────────────────────────────

def page_monitoring():
    st.markdown('<div class="main-header">📈 Model Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System health and model performance tracking</div>', unsafe_allow_html=True)

    health = api_get("/health")
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status", "🟢 Healthy" if health["status"] == "healthy" else "🔴 Degraded")
        col2.metric("Model Loaded", "✅ Yes" if health["model_loaded"] else "❌ No")
        col3.metric("Version", health["version"])
    else:
        st.error("Cannot reach API")
        return

    st.markdown("---")
    st.subheader("🧪 Run Inference Benchmark")

    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5,
    }

    n_runs = st.slider("Number of inference runs", 1, 50, 10)

    if st.button("▶️ Run Benchmark"):
        times = []
        probs = []
        progress = st.progress(0)
        for i in range(n_runs):
            t0 = time.time()
            r = api_post("/predict", sample_customer)
            elapsed = (time.time() - t0) * 1000
            if r:
                times.append(elapsed)
                probs.append(r["churn_probability"])
            progress.progress((i + 1) / n_runs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Inference (ms)", f"{sum(times)/len(times):.1f}")
        col2.metric("P95 Latency (ms)", f"{sorted(times)[int(len(times)*0.95)]:.1f}")
        col3.metric("Prediction Runs", n_runs)
        col4.metric("Avg Prob", f"{sum(probs)/len(probs):.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=times, mode="lines+markers", name="Latency (ms)"))
        fig.update_layout(title="Inference Latency Over Runs", xaxis_title="Run #", yaxis_title="Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("## 🧠 Revenue Risk AI")
        st.markdown(f"**API:** `{API_URL}`")

        health = api_get("/health")
        if health and health.get("status") == "healthy":
            st.success("● API Connected")
        else:
            st.error("● API Offline")

        st.markdown("---")
        pages = {
            "🎯 Single Customer Analysis": page_single_customer,
            "📁 Batch CSV Upload": page_batch_upload,
            "💰 Revenue Dashboard": page_revenue_dashboard,
            "🔬 Explainability": page_explainability,
            "🎯 Retention Strategy": page_retention,
            "📈 Model Monitoring": page_monitoring,
        }
        choice = st.radio("Navigation", list(pages.keys()))
        st.markdown("---")
        st.caption("AI Revenue Risk Intelligence v1.0")
        st.caption("Powered by LightGBM + SHAP + Groq")

    pages[choice]()


if __name__ == "__main__":
    main()
