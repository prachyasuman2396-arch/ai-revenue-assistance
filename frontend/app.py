import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ==============================================================================
# Page Configuration & Executive Theme
# ==============================================================================
st.set_page_config(
    page_title="AI Revenue Assistance | Executive Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-label {
        font-size: 0.825rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .email-container {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        color: #e2e8f0;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Endpoint Configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

# ==============================================================================
# Sidebar & Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&q=80", use_container_width=True)
    st.title("Revenue Defense AI")
    st.caption("Production MLOps & LLM Retention Engine")

    api_url = st.text_input("Backend API Gateway", value=DEFAULT_API_URL)

    # Health Check Probe
    api_online = False
    model_name_tag = "Unavailable"
    model_version_tag = "N/A"
    try:
        r_health = requests.get(f"{api_url}/api/health", timeout=2)
        if r_health.status_code == 200:
            api_online = True
            h_data = r_health.json()
            model_name_tag = h_data.get("model_name", "ChurnPredictor")
            model_version_tag = str(h_data.get("model_version", "1"))
    except Exception:
        pass

    if api_online:
        st.success(f"🟢 API Online | Model: {model_name_tag} v{model_version_tag}")
    else:
        st.error(f"🔴 API Offline ({api_url})")

    st.divider()
    st.markdown("### Quick Preset Scenarios")
    preset_choice = st.radio(
        "Load Customer Profile:",
        [
            "Custom Telemetry",
            "🚨 High Flight Risk (Month-to-Month, Fiber)",
            "🛡️ High Loyalty Safe (2-Year, Bundled)",
            "⚠️ Moderate Risk Onboarding (Tech Friction)",
        ],
    )

# ==============================================================================
# Presets Data Mapping
# ==============================================================================
default_customer = {
    "customer_id": "CUST_LIVE_902",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": False,
    "dependents": False,
    "tenure": 2,
    "phone_service": True,
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": True,
    "payment_method": "Electronic check",
    "monthly_charges": 95.50,
    "total_charges": 191.00,
}

if preset_choice == "🚨 High Flight Risk (Month-to-Month, Fiber)":
    default_customer.update({
        "customer_id": "CHURN_RISK_HIGH_01",
        "tenure": 1,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "online_security": "No",
        "monthly_charges": 99.80,
        "total_charges": 99.80,
        "payment_method": "Electronic check",
    })
elif preset_choice == "🛡️ High Loyalty Safe (2-Year, Bundled)":
    default_customer.update({
        "customer_id": "LOYAL_ENTERPRISE_88",
        "tenure": 64,
        "contract": "Two year",
        "internet_service": "DSL",
        "tech_support": "Yes",
        "online_security": "Yes",
        "online_backup": "Yes",
        "monthly_charges": 65.00,
        "total_charges": 4160.00,
        "payment_method": "Bank transfer (automatic)",
    })
elif preset_choice == "⚠️ Moderate Risk Onboarding (Tech Friction)":
    default_customer.update({
        "customer_id": "MOD_RISK_ONBOARD_14",
        "tenure": 5,
        "contract": "One year",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "monthly_charges": 85.00,
        "total_charges": 425.00,
    })

# ==============================================================================
# Header & Navigation Tabs
# ==============================================================================
st.title("🛡️ AI Revenue Risk Intelligence Platform")
st.markdown(
    "Real-time customer churn interception, financial exposure valuation, and automated Groq LLM retention playbooks."
)

tab_simulator, tab_db_explorer, tab_arch = st.tabs([
    "🚀 Real-Time Risk Simulator",
    "🗄️ Customer Accounts & Audit Trail",
    "🏛️ System Architecture & MLOps",
])

# ==============================================================================
# TAB 1: Real-Time Risk Simulator
# ==============================================================================
with tab_simulator:
    col_input, col_results = st.columns([1.1, 1.9], gap="large")

    with col_input:
        st.subheader("Customer Telemetry Input")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                cust_id = st.text_input("Customer ID", value=default_customer["customer_id"])
                tenure = st.slider("Tenure (Months)", 0, 72, int(default_customer["tenure"]))
                contract = st.selectbox(
                    "Contract Type",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(default_customer["contract"]),
                )
                internet = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"],
                    index=["DSL", "Fiber optic", "No"].index(default_customer["internet_service"]),
                )
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(
                        default_customer["payment_method"]
                    ),
                )
            with c2:
                monthly = st.number_input("Monthly Charges ($)", 10.0, 300.0, float(default_customer["monthly_charges"]), step=1.0)
                total = st.number_input("Total Lifetime Billed ($)", 10.0, 15000.0, float(default_customer["total_charges"]), step=10.0)
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["tech_support"]))
                security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["online_security"]))
                paperless = st.checkbox("Paperless Billing", value=bool(default_customer["paperless_billing"]))

            with st.expander("Additional Telemetry Features"):
                e1, e2 = st.columns(2)
                with e1:
                    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_customer["gender"] == "Female" else 1)
                    senior = st.checkbox("Senior Citizen", value=bool(default_customer["senior_citizen"]))
                    partner = st.checkbox("Has Partner", value=bool(default_customer["partner"]))
                    dependents = st.checkbox("Has Dependents", value=bool(default_customer["dependents"]))
                with e2:
                    phone = st.checkbox("Phone Service", value=bool(default_customer["phone_service"]))
                    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_customer["multiple_lines"]))
                    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["online_backup"]))
                    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["device_protection"]))
                    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["streaming_tv"]))
                    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_customer["streaming_movies"]))

            submit_btn = st.form_submit_button("⚡ Evaluate Risk & Generate Retention Strategy", use_container_width=True)

    with col_results:
        st.subheader("Intelligence Output & Prescriptive Action")

        if submit_btn:
            payload = {
                "customer_id": cust_id,
                "gender": gender,
                "senior_citizen": 1 if senior else 0,
                "partner": partner,
                "dependents": dependents,
                "tenure": tenure,
                "phone_service": phone,
                "multiple_lines": multiple,
                "internet_service": internet,
                "online_security": security,
                "online_backup": backup,
                "device_protection": protection,
                "tech_support": tech_support,
                "streaming_tv": tv,
                "streaming_movies": movies,
                "contract": contract,
                "paperless_billing": paperless,
                "payment_method": payment,
                "monthly_charges": monthly,
                "total_charges": total,
            }

            with st.spinner("🧠 Querying MLflow Model & Synthesizing Retention Playbook via Groq LLM..."):
                t_start = time.time()
                try:
                    resp = requests.post(f"{api_url}/api/predict", json=payload, timeout=20)
                    t_elapsed = round(time.time() - t_start, 2)

                    if resp.status_code == 200:
                        data = resp.json()

                        churn_prob = data["churn_probability"]
                        risk_info = data["risk_analysis"]
                        rec = data["recommendation"]
                        expl = data["explanation"]
                        risk_cat = risk_info["risk_category"]

                        # Top Metric Strip
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Flight Probability</div>
                                    <div class="metric-val" style="color: {'#ef4444' if churn_prob > 0.6 else '#f59e0b' if churn_prob > 0.35 else '#10b981'}">
                                        {round(churn_prob * 100, 1)}%
                                    </div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with m2:
                            badge_cls = "badge-high" if risk_cat == "HIGH" else "badge-med" if risk_cat == "MODERATE" else "badge-low"
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Risk Category</div>
                                    <div class="metric-val"><span class="{badge_cls}">{risk_cat} RISK</span></div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with m3:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">6-Mo Revenue At Risk</div>
                                    <div class="metric-val" style="color: #f87171;">${risk_info['revenue_at_risk']:,.2f}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with m4:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Annual Exposure</div>
                                    <div class="metric-val" style="color: #cbd5e1;">${risk_info['annual_revenue_exposure']:,.2f}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        # Row 2: Gauge & Explainability
                        g_col, expl_col = st.columns([1, 1.2])

                        with g_col:
                            fig = go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=churn_prob * 100,
                                    domain={"x": [0, 1], "y": [0, 1]},
                                    title={"text": "Flight Risk Index", "font": {"size": 16, "color": "#f8fafc"}},
                                    number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
                                    gauge={
                                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                                        "bar": {"color": "#ef4444" if churn_prob > 0.6 else "#f59e0b" if churn_prob > 0.35 else "#10b981"},
                                        "steps": [
                                            {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                                            {"range": [35, 60], "color": "rgba(245, 158, 11, 0.2)"},
                                            {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"},
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 3},
                                            "thickness": 0.75,
                                            "value": 60,
                                        },
                                    },
                                )
                            )
                            fig.update_layout(
                                height=240,
                                margin=dict(l=20, r=20, t=30, b=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={"color": "#f8fafc"},
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with expl_col:
                            st.markdown("#### 🔍 Model Factor Contributions")
                            st.caption(expl.get("summary", ""))

                            # Build Visual Drivers
                            risk_factors = expl.get("top_risk_factors", [])
                            prot_factors = expl.get("top_protective_factors", [])

                            if risk_factors:
                                st.markdown("**Escalating Risk Drivers (Pushing Score Up):**")
                                for rf in risk_factors:
                                    desc = rf.get("description") or rf.get("business_insight") or "Contributes to flight probability."
                                    score = f" (+{rf['attribution_score']})" if "attribution_score" in rf else ""
                                    st.markdown(f"🔴 **{rf.get('feature', 'Factor')}**{score}: {desc}")

                            if prot_factors:
                                st.markdown("**Protective Factors (Shielding Account):**")
                                for pf in prot_factors:
                                    desc = pf.get("description") or pf.get("business_insight") or "Fosters customer loyalty."
                                    score = f" ({pf['attribution_score']})" if "attribution_score" in pf else ""
                                    st.markdown(f"🟢 **{pf.get('feature', 'Factor')}**{score}: {desc}")

                        # Row 3: Prescriptive Retention Strategy
                        st.divider()
                        st.markdown(f"### 📋 Strategic Action Playbook: `{rec['playbook_code']}`")
                        
                        r_col1, r_col2 = st.columns([1.2, 1])
                        with r_col1:
                            st.info(f"**Primary Intervention:**\n\n{rec['primary_action']}")
                            st.markdown(f"**Estimated ROI Impact:** {rec['estimated_roi_impact']}")
                            if rec.get("secondary_actions"):
                                st.markdown("**Secondary Tactical Interventions:**")
                                for act in rec["secondary_actions"]:
                                    st.markdown(f"- {act}")

                        with r_col2:
                            p_out = rec.get("personalized_outreach")
                            if p_out:
                                st.markdown("#### 🤖 AI-Personalized Account Outreach")
                                st.markdown(f"**Subject:** `{p_out.get('retention_email_subject', 'Account Notice')}`")
                                
                                email_body = p_out.get("retention_email_body", "")
                                st.markdown(f"""<div class="email-container">{email_body}</div>""", unsafe_allow_html=True)
                                
                                talking_points = p_out.get("call_script_talking_points", [])
                                if talking_points:
                                    with st.expander("📞 CS Call Script & Talking Points"):
                                        for tp in talking_points:
                                            st.markdown(f"- {tp}")
                                
                                objection = p_out.get("counter_objection_strategy")
                                if objection:
                                    with st.expander("🛡️ Objection Counter-Strategy Battlecard"):
                                        st.write(objection)

                        st.caption(f"⚡ End-to-end inference + LLM orchestration completed in {t_elapsed}s | Powered by Groq + LangChain")

                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"Failed to connect to backend: {str(ex)}")
        else:
            st.info("👈 Select a customer preset on the left or customize parameters, then click **Evaluate Risk** to generate live predictions!")

# ==============================================================================
# TAB 2: Customer Accounts & Audit Explorer
# ==============================================================================
with tab_db_explorer:
    st.subheader("PostgreSQL Account Database & Retention Audit Trail")
    st.caption("Browse stored telemetry or trigger real-time evaluation for existing subscribers.")

    try:
        r_cust = requests.get(f"{api_url}/api/customers?limit=25", timeout=5)
        if r_cust.status_code == 200:
            customers_list = r_cust.json()
            if customers_list:
                df_cust = pd.DataFrame(customers_list)
                st.dataframe(
                    df_cust[["customer_id", "tenure", "contract", "internet_service", "monthly_charges", "payment_method", "churn"]],
                    use_container_width=True,
                )

                selected_cust = st.selectbox("Select Customer to Evaluate from Database:", df_cust["customer_id"].tolist())
                if st.button(f"🔍 Evaluate Existing Customer '{selected_cust}'"):
                    with st.spinner(f"Evaluating {selected_cust} from PostgreSQL..."):
                        r_eval = requests.post(f"{api_url}/api/predict/{selected_cust}", timeout=15)
                        if r_eval.status_code == 200:
                            st.success(f"Audit record persisted to PostgreSQL for '{selected_cust}'!")
                            st.json(r_eval.json())
                        else:
                            st.error(f"Failed: {r_eval.text}")
            else:
                st.info("No raw customer records found in PostgreSQL. Seed database via `python scripts/seed_database.py`.")
        else:
            st.warning(f"Could not reach database endpoint: {r_cust.text}")
    except Exception as ex:
        st.error(f"Error connecting to API: {str(ex)}")

# ==============================================================================
# TAB 3: System Architecture & MLOps Health
# ==============================================================================
with tab_arch:
    st.subheader("Production System Architecture")
    st.markdown(
        """
        ```mermaid
        graph TD
            User((Subscriber Telemetry)) --> FastAPIGateway[FastAPI Gateway :8000]
            FastAPIGateway --> MLflowRegistry[MLflow Model Registry :5001]
            FastAPIGateway --> PostgresDB[(PostgreSQL Database :5433)]
            FastAPIGateway --> Explainer[SHAP / Linear Feature Contribution Engine]
            FastAPIGateway --> GroqLLM[Groq LLM Engine (openai/gpt-oss-120b)]
            GroqLLM --> StructuredOutreach[Personalized Retention Email & Battlecard]
            FastAPIGateway --> StreamlitUI[Streamlit Executive Dashboard :8501]
        ```
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔌 API Documentation")
        st.write("Interactive OpenAPI / Swagger UI:")
        st.markdown(f"👉 [{api_url}/docs]({api_url}/docs)")

    with c2:
        st.markdown("### 🧪 MLflow Registry")
        st.write("Experiment tracking & artifact lineage:")
        st.markdown("👉 [http://localhost:5001](http://localhost:5001)")

    with c3:
        st.markdown("### 🐘 PostgreSQL Store")
        st.write("Stores telemetry, prediction logs, and audit trails on port `5433`.")
