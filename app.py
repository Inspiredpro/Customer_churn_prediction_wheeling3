"""
Customer Churn Risk Predictor - Streamlit app
=============================================
Serves the Logistic Regression model from the Customer Churn notebook.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> https://share.streamlit.io -> select app.py

Put your saved model (.joblib / .pkl) next to this file or in model/. The model must have been trained on the notebook's
10 features, one-hot encoded with pd.get_dummies (Step 9).
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration taken from the notebook (Steps 8-9 and the Predictor v1 cell)
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORIES = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "PaperlessBilling": ["No", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
}

# "Typical customer" defaults used in the notebook (median / most common value)
DEFAULTS = {
    "tenure": 29,
    "MonthlyCharges": 70.35,
    "SeniorCitizen": 0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "TechSupport": "No",
    "OnlineSecurity": "No",
}

# Column order the notebook's pd.get_dummies() produced. Used only if the saved
# model does not carry its own feature_names_in_.
FALLBACK_COLUMNS = NUMERIC_FEATURES + [
    f"{col}_{cat}" for col in CATEGORICAL_FEATURES for cat in CATEGORIES[col]
]

APP_DIR = Path(__file__).resolve().parent
MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle", ".sav"}
SEARCH_DIRS = [APP_DIR, APP_DIR / "model", APP_DIR / "models"]

COLORS = {
    "ink": "#1B2430",
    "muted": "#4A5565",
    "accent": "#0F6E8C",
    "low": "#1F7A57",
    "medium": "#A86A00",
    "high": "#C0322A",
}

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, label {
        font-family: 'Figtree', 'Segoe UI', sans-serif;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    .subtitle { color: #5B6675; font-size: 1.05rem; margin-bottom: 1.4rem; max-width: 62ch; }

    .verdict {
        background: #FFFFFF; border: 1px solid #D5DAE1; border-left: 8px solid var(--c);
        border-radius: 10px; padding: 1rem 1.25rem; margin-top: -0.5rem; color: #1B2430 !important;
    }
    .verdict .level { font-size: 1.6rem; font-weight: 700; color: var(--c) !important; line-height: 1.1; }
    .verdict .detail { color: #1B2430 !important; margin-top: 0.35rem; font-size: 0.98rem; }

    .action {
        background: #FFFFFF; border: 1px solid #D5DAE1; border-radius: 10px;
        padding: 0.7rem 1rem; margin-bottom: 0.55rem; font-size: 0.95rem;
        color: #1B2430 !important;
    }
    .action, .action span, .action p { color: #1B2430 !important; }
    .action b { color: #0B5870 !important; }
    .section-title { font-size: 1.05rem; font-weight: 600; margin: 0.6rem 0 0.3rem 0; }

    /* Metric cards: white background, so force dark text for label and value */
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #D5DAE1; border-radius: 10px; padding: 0.7rem 1rem;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: #1B2430 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def find_model_files() -> tuple[Path | None, Path | None]:
    """Find the model (and an optional scaler) in the app folder or model/ subfolders."""
    found: list[Path] = []
    for folder in SEARCH_DIRS:
        if folder.is_dir():
            found += sorted(p for p in folder.iterdir() if p.suffix.lower() in MODEL_EXTENSIONS)
    scalers = [p for p in found if "scaler" in p.stem.lower()]
    models = [p for p in found if p not in scalers]
    # Prefer a file whose name mentions churn / model / logistic
    models.sort(key=lambda p: not any(k in p.stem.lower() for k in ("churn", "model", "logistic", "logreg")))
    return (models[0] if models else None), (scalers[0] if scalers else None)


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Load the saved model (and an optional separate scaler)."""
    model_path, scaler_path = find_model_files()
    if model_path is None:
        return None, None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler, model_path.name


def expected_columns(model, scaler) -> list[str]:
    """Training columns, in order. Prefer what the model itself remembers."""
    for obj in (model, getattr(model, "steps", [[None, None]])[0][1], scaler):
        cols = getattr(obj, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    return FALLBACK_COLUMNS


def linear_parts(model, scaler):
    """Return (scaler, linear_model) so we can explain predictions, if possible."""
    if hasattr(model, "steps"):  # sklearn Pipeline
        steps = [s for _, s in model.steps]
        lin = steps[-1]
        sc = next((s for s in steps[:-1] if hasattr(s, "scale_")), None)
        return sc, lin
    return scaler, model


# ----------------------------------------------------------------------------
# Data preparation (mirrors the notebook's cleaning + encoding)
# ----------------------------------------------------------------------------
def encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode exactly like Step 9 and align to the training columns."""
    work = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        work[col] = pd.Categorical(work[col], categories=CATEGORIES[col])
    work = pd.get_dummies(work, columns=CATEGORICAL_FEATURES, dtype=int)
    return work.reindex(columns=columns, fill_value=0)


def predict_proba(df: pd.DataFrame, model, scaler, columns) -> np.ndarray:
    X = encode(df, columns)
    if scaler is not None and not hasattr(model, "steps"):
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


def risk_level(p: float, high: float, medium: float) -> str:
    if p >= high:
        return "HIGH"
    if p >= medium:
        return "MEDIUM"
    return "LOW"


def explain(row: pd.DataFrame, model, scaler, columns) -> pd.Series | None:
    """Per-feature push on the churn log-odds relative to an average customer."""
    sc, lin = linear_parts(model, scaler)
    if not hasattr(lin, "coef_"):
        return None
    X = encode(row, columns)
    Xs = sc.transform(X) if sc is not None else X.to_numpy(dtype=float)
    contrib = pd.Series(Xs[0] * lin.coef_[0], index=columns)

    grouped: dict[str, float] = {}
    for col, val in contrib.items():
        base = next((f for f in FEATURES if col == f or col.startswith(f + "_")), col)
        grouped[base] = grouped.get(base, 0.0) + float(val)
    return pd.Series(grouped).sort_values()


# ----------------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------------
LEVEL_COLOR = {"LOW": COLORS["low"], "MEDIUM": COLORS["medium"], "HIGH": COLORS["high"]}
LEVEL_TEXT = {
    "LOW": "Unlikely to leave. Keep the relationship healthy.",
    "MEDIUM": "Worth a proactive check-in before it gets worse.",
    "HIGH": "Likely to leave. Prioritise this customer for retention outreach.",
}


def gauge(p: float, high: float, medium: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(p * 100, 1),
            number={"suffix": "%", "font": {"size": 46, "color": COLORS["ink"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6B7684", "tickfont": {"color": COLORS["ink"]}},
                "bar": {"color": COLORS["ink"], "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, medium * 100], "color": "#CFE8DF"},
                    {"range": [medium * 100, high * 100], "color": "#F6E2B3"},
                    {"range": [high * 100, 100], "color": "#F0C3BF"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="#FFFFFF",
        font={"family": "Figtree, sans-serif", "color": COLORS["ink"]},
    )
    return fig


def drivers_chart(contrib: pd.Series) -> go.Figure:
    labels = {
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "SeniorCitizen": "Senior citizen",
        "Contract": "Contract type",
        "InternetService": "Internet service",
        "PaymentMethod": "Payment method",
        "PaperlessBilling": "Paperless billing",
        "TechSupport": "Tech support",
        "OnlineSecurity": "Online security",
    }
    c = contrib.copy()
    c.index = [labels.get(i, i) for i in c.index]
    c = c.reindex(c.abs().sort_values().index)  # biggest effect on top
    fig = go.Figure(
        go.Bar(
            x=c.values,
            y=c.index,
            orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in c.values],
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360, margin=dict(l=0, r=10, t=10, b=30),
        xaxis_title="← lowers risk      raises risk →",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font={"family": "Figtree, sans-serif", "color": COLORS["ink"]},
        xaxis=dict(zeroline=True, zerolinecolor="#6B7684", gridcolor="#E7EAEE",
                   tickfont=dict(color=COLORS["ink"]), title_font=dict(color=COLORS["ink"])),
        yaxis=dict(tickfont=dict(color=COLORS["ink"])),
    )
    return fig


def suggested_actions(c: dict) -> list[tuple[str, str]]:
    """Plain-language retention ideas based on the patterns found in the notebook."""
    out = []
    if c["Contract"] == "Month-to-month":
        out.append(("Offer a longer contract", "Month-to-month customers churn far more than annual or two-year ones. A small discount for a 1-year plan can pay for itself."))
    if c["tenure"] <= 12:
        out.append(("Run an onboarding check-in", "Most churn happens early. A call or message in the first months builds the habit of staying."))
    if c["InternetService"] == "Fiber optic" and c["TechSupport"] == "No":
        out.append(("Bundle free tech support", "Fiber customers without tech support are a high-churn group. Add support for a trial period."))
    if c["OnlineSecurity"] == "No" and c["InternetService"] != "No":
        out.append(("Offer online security", "Customers without it leave more often. A free trial raises engagement."))
    if c["PaymentMethod"] == "Electronic check":
        out.append(("Encourage automatic payment", "Electronic-check payers show the highest churn. Offer a small credit to switch to card or bank autopay."))
    if c["MonthlyCharges"] >= 85:
        out.append(("Review the plan price", "High monthly bills are linked to churn. Check whether a cheaper plan fits their usage."))
    return out


def customer_frame(c: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: c[k] for k in FEATURES}])


# ----------------------------------------------------------------------------
# Header + guard
# ----------------------------------------------------------------------------
st.title("Customer Churn Risk Predictor")
st.markdown(
    '<div class="subtitle">Estimate how likely a telecom customer is to leave, see what is driving '
    "the risk, and decide who the retention team should call first.</div>",
    unsafe_allow_html=True,
)

model, scaler, model_path = load_artifacts()
if model is None:
    seen = sorted(p.name for p in APP_DIR.iterdir() if p.is_file())
    st.error(
        "No model file found. Add your saved logistic regression (`.joblib` or `.pkl`) "
        "to the same folder as `app.py`, commit it to GitHub, then reboot the app."
    )
    st.caption("Files the app can see next to app.py: " + ", ".join(seen))
    st.stop()

COLUMNS = expected_columns(model, scaler)

# ----------------------------------------------------------------------------
# Sidebar: business settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Risk settings")
    st.caption("The model outputs a probability. These cut-offs turn it into a risk level.")
    high_t = st.slider("HIGH risk at or above", 0.10, 0.90, 0.50, 0.05, format="%.2f")
    med_t = st.slider("MEDIUM risk at or above", 0.05, 0.85, 0.30, 0.05, format="%.2f")
    if med_t >= high_t:
        st.warning("MEDIUM cut-off should be lower than HIGH. Adjusted automatically.")
        med_t = max(0.05, high_t - 0.05)
    st.caption(
        "Lower cut-offs catch more churners but create more false alarms. "
        "That is a business choice: how costly is a missed customer versus an unnecessary call?"
    )
    st.divider()
    st.caption(f"Model file: `{model_path}`")

tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Score a customer list", "About"])

# ----------------------------------------------------------------------------
# Tab 1: single customer
# ----------------------------------------------------------------------------
with tab_single:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Account</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        tenure = a1.slider("Tenure (months)", 0, 72, DEFAULTS["tenure"])
        monthly = a2.number_input("Monthly charges ($)", 15.0, 125.0, float(DEFAULTS["MonthlyCharges"]), 0.5)

        auto_total = st.checkbox("Estimate total charges as tenure × monthly (as in the notebook)", value=True)
        if auto_total:
            total = float(tenure * monthly)
            st.caption(f"Total charges used: ${total:,.2f}")
        else:
            total = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure * monthly), 10.0)

        c1, c2 = st.columns(2)
        contract = c1.selectbox("Contract", CATEGORIES["Contract"], index=CATEGORIES["Contract"].index(DEFAULTS["Contract"]))
        payment = c2.selectbox("Payment method", CATEGORIES["PaymentMethod"], index=CATEGORIES["PaymentMethod"].index(DEFAULTS["PaymentMethod"]))
        c3, c4 = st.columns(2)
        paperless = c3.radio("Paperless billing", CATEGORIES["PaperlessBilling"], index=1, horizontal=True)
        senior = c4.radio("Senior citizen", ["No", "Yes"], index=0, horizontal=True)

        st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)
        internet = st.selectbox("Internet service", CATEGORIES["InternetService"], index=CATEGORIES["InternetService"].index(DEFAULTS["InternetService"]))
        if internet == "No":
            tech = security = "No internet service"
            st.caption("No internet service, so tech support and online security are set to “No internet service”.")
        else:
            s1, s2 = st.columns(2)
            tech = s1.radio("Tech support", ["No", "Yes"], index=0, horizontal=True)
            security = s2.radio("Online security", ["No", "Yes"], index=0, horizontal=True)

    customer = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "TechSupport": tech,
        "OnlineSecurity": security,
    }
    prob = float(predict_proba(customer_frame(customer), model, scaler, COLUMNS)[0])
    level = risk_level(prob, high_t, med_t)

    with right:
        st.plotly_chart(gauge(prob, high_t, med_t), width="stretch", config={"displayModeBar": False})
        st.markdown(
            f'<div class="verdict" style="--c:{LEVEL_COLOR[level]}">'
            f'<div class="level">{level} RISK</div>'
            f'<div class="detail">{prob*100:.1f}% chance of churn. {LEVEL_TEXT[level]}</div></div>',
            unsafe_allow_html=True,
        )

        # What-if: change the contract (the notebook's own discussion question)
        st.markdown('<div class="section-title">What if we changed the contract?</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, option in zip(cols, CATEGORIES["Contract"]):
            alt = {**customer, "Contract": option}
            p_alt = float(predict_proba(customer_frame(alt), model, scaler, COLUMNS)[0])
            delta = (p_alt - prob) * 100
            col.metric(option, f"{p_alt*100:.1f}%", None if option == contract else f"{delta:+.1f} pts", delta_color="inverse")

    st.divider()
    d_col, a_col = st.columns([1.1, 1], gap="large")

    with d_col:
        st.markdown('<div class="section-title">What is driving this score</div>', unsafe_allow_html=True)
        contrib = explain(customer_frame(customer), model, scaler, COLUMNS)
        if contrib is None:
            st.info("Driver breakdown is available for linear models such as Logistic Regression.")
        else:
            st.plotly_chart(drivers_chart(contrib), width="stretch", config={"displayModeBar": False})
            st.caption("Compared with an average customer. Red bars push the risk up, green bars pull it down.")

    with a_col:
        st.markdown('<div class="section-title">Suggested retention actions</div>', unsafe_allow_html=True)
        actions = suggested_actions(customer)
        if level == "LOW" and not actions:
            st.success("No special action needed for this customer.")
        elif not actions:
            st.info("No rule-based suggestion applies. Consider a personal check-in.")
        for title, why in actions:
            st.markdown(f'<div class="action"><b>{title}.</b> {why}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab 2: batch scoring (the notebook's "retention list")
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV and get a ranked **retention list**, highest risk first.")
    st.caption(
        "Required columns: " + ", ".join(FEATURES) + ". The original Telco CSV works as is. "
        "Blank `TotalCharges` values (new customers) are handled like in the notebook."
    )

    template = pd.DataFrame([{**{k: DEFAULTS.get(k) for k in FEATURES}, "TotalCharges": round(29 * 70.35, 2)}])
    st.download_button("Download a template CSV", template.to_csv(index=False), "churn_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload customers (CSV)", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in raw.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                df = raw.copy()
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
                unknown = {
                    c: sorted(set(df[c].dropna()) - set(CATEGORIES[c]))
                    for c in CATEGORICAL_FEATURES
                    if set(df[c].dropna()) - set(CATEGORIES[c])
                }
                if unknown:
                    st.warning(f"Unrecognised category values will be treated as all-zero: {unknown}")

                probs = predict_proba(df, model, scaler, COLUMNS)
                result = raw.copy()
                result["ChurnRisk%"] = (probs * 100).round(1)
                result["RiskLevel"] = [risk_level(p, high_t, med_t) for p in probs]
                result = result.sort_values("ChurnRisk%", ascending=False).reset_index(drop=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Customers scored", f"{len(result):,}")
                m2.metric("High risk", f"{(result.RiskLevel == 'HIGH').sum():,}")
                m3.metric("Medium risk", f"{(result.RiskLevel == 'MEDIUM').sum():,}")
                m4.metric("Average risk", f"{result['ChurnRisk%'].mean():.1f}%")

                top_n = st.slider("Show the top N customers", 5, min(500, len(result)), min(25, len(result)))
                st.dataframe(
                    result.head(top_n),
                    width="stretch",
                    column_config={
                        "ChurnRisk%": st.column_config.ProgressColumn("ChurnRisk%", min_value=0, max_value=100, format="%.1f%%")
                    },
                )
                st.download_button(
                    "Download full ranked list",
                    result.to_csv(index=False),
                    "retention_list.csv",
                    "text/csv",
                    type="primary",
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")

# ----------------------------------------------------------------------------
# Tab 3: about
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### How this works
        The model is a **Logistic Regression** trained on the Telco Customer Churn dataset
        (about 7,000 customers). Inputs are scaled, text columns are one-hot encoded, and the
        model returns the **probability** that a customer will leave.

        **Features used:** tenure, monthly charges, total charges, senior citizen, contract,
        internet service, payment method, paperless billing, tech support, online security.

        ### Reading the results
        - **Risk level** comes from the cut-offs in the sidebar (default: HIGH ≥ 50%, MEDIUM ≥ 30%).
        - **Drivers** show how each feature moves the score compared with an average customer.
        - **Actions** are rules of thumb drawn from the patterns in the notebook (contract type,
          tenure, payment method, internet service), not model outputs.

        ### Limits to keep in mind
        - Only about a quarter of customers churn, so **accuracy alone is misleading**. Judge the
          model by how many real churners it catches (recall) and how many alarms are false (precision).
        - Correlation is not causation. A driver shows a pattern in past data, not a guaranteed cause.
        - The model only knows the 10 features above, so use it to rank and prioritise, not as a final verdict.
        """
    )
