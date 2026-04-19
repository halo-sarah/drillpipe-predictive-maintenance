# app.py
# Run: streamlit run app.py
# Pipe Integrity Monitoring — Coating Prediction & RUL Analytics

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services import (
    DrillPipeLifeEstimator,
    DEFAULT_JOB_STEPS,
    DEFAULT_COATING_SPECS,
    clamp,
    ds1_current_grade,
    ds1_next_drop,
)
from rules import compute_integrity

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pipe Integrity Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling — restrained, data-first, industrial
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f4f5f6;
        color: #1a1d23;
    }

    /* Header strip */
    .pim-header {
        background-color: #1a1d23;
        color: #f4f5f6;
        padding: 14px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        border-bottom: 3px solid #2f6fad;
    }
    .pim-header-title { font-size: 15px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
    .pim-header-sub   { font-size: 12px; font-weight: 300; color: #8a9bb0; letter-spacing: 0.04em; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #dde1e7;
        border-top: 3px solid #2f6fad;
        padding: 18px 20px 14px;
        margin-bottom: 0;
    }
    .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7280; margin-bottom: 4px; }
    .kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: #1a1d23; line-height: 1.1; }
    .kpi-unit  { font-size: 13px; font-weight: 400; color: #9ca3af; margin-left: 4px; }

    /* Status badge */
    .badge-normal   { background:#d1fae5; color:#065f46; padding:3px 10px; font-size:11px; font-weight:600; letter-spacing:0.06em; border-radius:2px; }
    .badge-warning  { background:#fef3c7; color:#92400e; padding:3px 10px; font-size:11px; font-weight:600; letter-spacing:0.06em; border-radius:2px; }
    .badge-critical { background:#fee2e2; color:#991b1b; padding:3px 10px; font-size:11px; font-weight:600; letter-spacing:0.06em; border-radius:2px; }

    /* Section header */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        border-bottom: 1px solid #dde1e7;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Result table */
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th { text-align: left; padding: 6px 10px; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #6b7280; border-bottom: 1px solid #dde1e7; }
    .data-table td { padding: 8px 10px; border-bottom: 1px solid #f0f1f3; font-family: 'IBM Plex Mono', monospace; }
    .data-table tr:last-child td { border-bottom: none; }

    /* Recommendation panel */
    .rec-normal   { border-left: 4px solid #10b981; padding: 14px 18px; background: #f0fdf4; font-size: 14px; }
    .rec-warning  { border-left: 4px solid #f59e0b; padding: 14px 18px; background: #fffbeb; font-size: 14px; }
    .rec-critical { border-left: 4px solid #ef4444; padding: 14px 18px; background: #fef2f2; font-size: 14px; }
    .rec-label    { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }

    /* Run button */
    div.stButton > button {
        background-color: #1a1d23;
        color: #f4f5f6;
        border: none;
        border-radius: 2px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.6em 1.8em;
        width: 100%;
    }
    div.stButton > button:hover { background-color: #2f6fad; }

    /* Plotly chart wrapper */
    .chart-container { background: #ffffff; border: 1px solid #dde1e7; padding: 4px; }

    /* Suppress Streamlit chrome noise */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="pim-header">
    <div>
        <div class="pim-header-title">Pipe Integrity Monitoring System</div>
        <div class="pim-header-sub">Coating Classification · Corrosion Rate · Remaining Useful Life</div>
    </div>
    <div class="pim-header-sub">Engineering Analytics v2.0 — 2026</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    rf  = joblib.load("model_rf_coating.pkl")
    svm = joblib.load("model_svm_coating.pkl")
    scaler = joblib.load("scaler_coating.pkl")
    le = joblib.load("label_encoder_coating.pkl")
    return rf, svm, scaler, le

try:
    model_rf, model_svm, scaler, label_enc = load_models()
    class_names = label_enc.classes_
except Exception as err:
    st.error(f"Model load failed: {err}")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensemble_predict(rf_p: np.ndarray, svm_p: np.ndarray, w_rf: float = 0.6) -> np.ndarray:
    return w_rf * rf_p + (1.0 - w_rf) * svm_p

def badge(status: str) -> str:
    cls = {"NORMAL": "badge-normal", "WARNING": "badge-warning", "CRITICAL": "badge-critical"}.get(status, "badge-normal")
    return f'<span class="{cls}">{status}</span>'


# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Operating Parameters — ML Coating Input</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    stage   = st.selectbox("Stage", ["I", "II", "III", "IV"], index=3)
    fluid   = st.selectbox("Fluid System", ["CLAYSTIM", "FLOWSTIM", "FLUSH"])
    flow_rate = st.number_input("Flow Rate (BPM)", value=3.0, format="%.1f")
    soak_time = st.number_input("Soak Time (hr)", value=2.0, format="%.1f")
with col2:
    temp_f    = st.number_input("Temperature (°F)", value=180.0, format="%.1f")
    total_vol = st.number_input("Total Volume (bbl)", value=29.0, format="%.1f")
    exposure  = st.number_input("Exposure Time (hr)", value=0.1, format="%.2f")
    hcl_pct   = st.number_input("HCl Concentration (%)", value=32.0, format="%.1f")
with col3:
    hf_acid   = st.number_input("HF Acid (GPT)", value=40.0, format="%.1f")
    abf_conc  = st.number_input("ABF Concentration (pptg)", value=250.0, format="%.1f")
    chelating = st.number_input("Chelating Agent (PPTG)", value=0.0, format="%.1f")
    sw_flush  = st.number_input("Seawater Flush (PSU)", value=36.25, format="%.2f")
with col4:
    sw_exp    = st.number_input("Seawater Exposure (hr)", value=0.5, format="%.2f")
    thick     = st.number_input("Coating Thickness (mils)", value=9.0, format="%.1f")
    acid_res  = st.number_input("Acid Resistance (hr)", value=80.0, format="%.1f")
    taber     = st.number_input("Taber Abrasion (mg/1000cyc)", value=67.0, format="%.1f")
    temp_res  = st.number_input("Temp Resistance (°F)", value=350.0, format="%.1f")

st.markdown('<div class="section-label" style="margin-top:24px;">Wall Thickness Inspection</div>', unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
with r1: wt_before  = st.number_input("WT Reference / New (in)", value=0.368, format="%.4f")
with r2: wt_current = st.number_input("WT Current Measured (in)", value=0.356, format="%.4f")
with r3: wt_min     = st.number_input("WT Minimum / Scrap (in)", value=0.310, format="%.4f")
with r4: alpha      = st.number_input("Confidence Penalty α", value=0.50, format="%.2f",
                                       help="Scales up loss-per-job when ML confidence is low. α=0 disables penalty.")

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([3, 1, 3])
with btn_col:
    run = st.button("Run Analysis")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if run:
    stage_map = {"I": 0, "II": 1, "III": 2, "IV": 3}
    fluid_map = {"CLAYSTIM": 0, "FLOWSTIM": 1, "FLUSH": 2}

    feature_row = {
        "Stage": stage_map[stage],
        "Fluid_System": fluid_map[fluid],
        "Flow_Rate_BPM": flow_rate,
        "Soak_Time_hr": soak_time,
        "Temperature_F": temp_f,
        "Total_Volume_bbl": total_vol,
        "Exposure_Time_hr": exposure,
        "HCl_Conc_pct": hcl_pct,
        "HF_Acid_Conc_GPT": hf_acid,
        "ABF_Conc_pptg": abf_conc,
        "Chelating_Agent_Conc_PPTG": chelating,
        "Seawater_Flush_PSU": sw_flush,
        "Seawater_Exposure_hr": sw_exp,
        "Coating_Thickness_mils": thick,
        "Acid_Resistance_hr_28pctHCL_200F_Consecutively": acid_res,
        "Taber_Abrasion_mg_1000cyc": taber,
        "Temperature_Resistance_F": temp_res,
    }
    df_in = pd.DataFrame([feature_row])

    # --- ML predictions ---
    rf_probs  = model_rf.predict_proba(df_in)[0]
    svm_probs = model_svm.predict_proba(scaler.transform(df_in))[0]
    ens_probs = ensemble_predict(rf_probs, svm_probs)

    best_coating = class_names[np.argmax(ens_probs)]
    confidence   = float(np.max(ens_probs))

    # --- RUL / corrosion ---
    estimator = DrillPipeLifeEstimator(
        od_in=5.0,
        wt_before_in=wt_before,
        wt_measured_in=wt_current,
        wt_min_in=wt_min,
        grade="G105",
        job_steps=DEFAULT_JOB_STEPS,
        coating_specs=DEFAULT_COATING_SPECS,
        jobs_per_year=12,
        k0_in_per_hour=2.5e-5,
        safety_factor=1.25,
    )
    base_loss   = estimator.loss_per_job_in(best_coating)
    risk_mult   = 1.0 + alpha * (1.0 - clamp(confidence, 0.0, 1.0))
    adj_loss    = base_loss * risk_mult
    jobs_remain = max((wt_current - wt_min) / adj_loss, 0.0) if adj_loss > 0 else float("inf")
    corr_rate   = adj_loss * 12 * 25.4   # mm/year

    grade_now                         = ds1_current_grade(wt_current, wt_before, wt_min)
    _, next_event, next_threshold_wt  = ds1_next_drop(wt_current, wt_before, wt_min)

    # --- Integrity score ---
    thickness_mm = wt_current * 25.4
    integrity    = compute_integrity(
        thickness_mm=thickness_mm,
        is_spike=False,
        spike_score=0.0,
        confidence=confidence,
        jobs_remaining=jobs_remain,
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # KPI cards
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-label">Condition Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def kpi(col, label, value, unit="", note=""):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
            {'<div style="font-size:12px;color:#9ca3af;margin-top:4px;">' + note + '</div>' if note else ''}
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Wall Thickness", f"{thickness_mm:.2f}", "mm", f"{wt_current:.4f} in")
    kpi(k2, "Integrity Score", f"{integrity.score:.1f}", "/ 100", integrity.status)
    kpi(k3, "Remaining Life", f"{jobs_remain:.1f}" if jobs_remain < 999 else "∞", "jobs", f"{corr_rate:.3f} mm/yr")
    kpi(k4, "DS-1 Grade", grade_now, "", next_event)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Maintenance recommendation
    # -------------------------------------------------------------------------
    rec_class = {"NORMAL": "rec-normal", "WARNING": "rec-warning", "CRITICAL": "rec-critical"}[integrity.status]
    st.markdown(f"""
    <div class="{rec_class}">
        <div class="rec-label">Maintenance Recommendation</div>
        <strong>{integrity.recommendation}</strong> &nbsp; {badge(integrity.status)}
        <div style="font-size:12px;margin-top:6px;color:#374151;">
            Integrity score {integrity.score:.1f}/100 &nbsp;·&nbsp;
            Confidence {confidence:.1%} &nbsp;·&nbsp;
            Risk multiplier {risk_mult:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Coating classification results
    # -------------------------------------------------------------------------
    left, right = st.columns([2, 3])

    with left:
        st.markdown('<div class="section-label">Coating Classification</div>', unsafe_allow_html=True)
        rows = []
        for i, name in enumerate(class_names):
            rows.append({
                "Coating": name,
                "RF": f"{rf_probs[i]*100:.1f}%",
                "SVM": f"{svm_probs[i]*100:.1f}%",
                "Ensemble": f"{ens_probs[i]*100:.1f}%",
            })
        df_table = pd.DataFrame(rows).sort_values("Ensemble", ascending=False)
        # Highlight predicted row
        table_html = '<table class="data-table"><thead><tr><th>Coating</th><th>RF</th><th>SVM</th><th>Ensemble</th></tr></thead><tbody>'
        for _, row in df_table.iterrows():
            bold = "font-weight:600;" if row["Coating"] == best_coating else ""
            table_html += f'<tr style="{bold}"><td>{row["Coating"]}</td><td>{row["RF"]}</td><td>{row["SVM"]}</td><td>{row["Ensemble"]}</td></tr>'
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:14px;font-size:13px;">
            <strong>Predicted:</strong> {best_coating} &nbsp;
            <span style="color:#6b7280;">({confidence:.1%} confidence)</span>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
        colors = {"TK34": "#1a1d23", "TK34P": "#2f6fad", "TC2000P": "#6b7280"}
        bar_colors = [colors.get(n, "#9ca3af") for n in class_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Ensemble", x=list(class_names), y=ens_probs * 100,
            marker_color=bar_colors, width=0.35,
        ))
        fig.add_trace(go.Bar(
            name="RF",  x=list(class_names), y=rf_probs * 100,
            marker_color=["#dde1e7"] * len(class_names), width=0.35,
        ))
        fig.update_layout(
            barmode="group",
            height=260,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="IBM Plex Sans", size=12, color="#1a1d23"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis=dict(title="Probability (%)", gridcolor="#f0f1f3", range=[0, 105]),
            xaxis=dict(gridcolor="#f0f1f3"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Corrosion / RUL detail
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-label">Corrosion & Remaining Useful Life Detail</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="data-table"><table class="data-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>', unsafe_allow_html=True)
        rul_rows = [
            ("Coating (predicted)", best_coating),
            ("Base loss / job", f"{base_loss:.6f} in"),
            ("Risk multiplier", f"{risk_mult:.4f}×"),
            ("Adjusted loss / job", f"{adj_loss:.6f} in"),
        ]
        html_r = '<table class="data-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>'
        for k, v in rul_rows:
            html_r += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html_r += "</tbody></table>"
        st.markdown(html_r, unsafe_allow_html=True)

    with d2:
        rul_rows2 = [
            ("Corrosion rate", f"{corr_rate:.3f} mm/yr"),
            ("Remaining life", f"{jobs_remain:.1f} jobs" if jobs_remain < 999 else "∞"),
            ("DS-1 grade", grade_now),
            ("Next event", next_event),
        ]
        html_r2 = '<table class="data-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>'
        for k, v in rul_rows2:
            html_r2 += f"<tr><td>{k}</td><td>{v}</td></tr>"
        if next_threshold_wt:
            html_r2 += f"<tr><td>Next event threshold</td><td>{next_threshold_wt:.4f} in</td></tr>"
        html_r2 += "</tbody></table>"
        st.markdown(html_r2, unsafe_allow_html=True)

    with d3:
        # Simple gauge-style bar for RUL
        life_pct = min(jobs_remain / 50.0, 1.0) * 100 if jobs_remain < 999 else 100
        color = "#10b981" if life_pct > 60 else ("#f59e0b" if life_pct > 25 else "#ef4444")
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #dde1e7;padding:16px;">
            <div class="kpi-label" style="margin-bottom:10px;">Life Remaining</div>
            <div style="background:#f0f1f3;height:14px;border-radius:2px;overflow:hidden;">
                <div style="background:{color};width:{life_pct:.0f}%;height:100%;"></div>
            </div>
            <div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;">
                {life_pct:.0f}<span style="font-size:13px;font-weight:400;color:#9ca3af;">%</span>
            </div>
            <div style="font-size:12px;color:#9ca3af;margin-top:2px;">of 50-job reference baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:16px;font-size:12px;color:#9ca3af;border-top:1px solid #f0f1f3;padding-top:10px;">
        Confidence penalty α modifies loss/job to account for ML model uncertainty. Higher α = more conservative RUL.
        DS-1 grades per API/ISO drill pipe inspection standard.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Enter operating parameters and wall thickness measurements, then click **Run Analysis**.")
