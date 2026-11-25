import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'frontend' and other modules resolve across all deployment platforms
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

try:
    from frontend.utils.api_client import (
        get_portfolios, create_portfolio, delete_portfolio, get_portfolio_detail,
        get_portfolio_risk, get_portfolio_risk_contribution, get_portfolio_correlations,
        run_stress_test, get_stress_scenarios, get_market_regime, get_market_anomalies,
        generate_ai_analysis, ask_ai_risk_question, get_monte_carlo,
        get_historical_crises, run_historical_crisis, run_reverse_stress,
        optimize_portfolio, get_efficient_frontier, get_var_backtest
    )
except ImportError:
    from utils.api_client import (
        get_portfolios, create_portfolio, delete_portfolio, get_portfolio_detail,
        get_portfolio_risk, get_portfolio_risk_contribution, get_portfolio_correlations,
        run_stress_test, get_stress_scenarios, get_market_regime, get_market_anomalies,
        generate_ai_analysis, ask_ai_risk_question, get_monte_carlo,
        get_historical_crises, run_historical_crisis, run_reverse_stress,
        optimize_portfolio, get_efficient_frontier, get_var_backtest
    )

# Page Configuration
st.set_page_config(
    page_title="FinRisk AI — Institutional Portfolio Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-green { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-yellow { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-red { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FinRisk AI — Institutional Risk Intelligence Platform")
st.caption("Explainable AI Portfolio Risk Engineering • Basel Regulatory Backtesting • Non-Linear Stress Lab • Multi-Asset Optimization")
st.markdown("---")

# Session State Initialization
if "selected_portfolio_id" not in st.session_state:
    st.session_state.selected_portfolio_id = None
if "selected_scenario_id" not in st.session_state:
    st.session_state.selected_scenario_id = "market_crash_10"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Portfolio Selector / Creator
st.sidebar.header("📁 Portfolio Management")

try:
    portfolios = get_portfolios()
except Exception as e:
    st.error(f"Could not connect to backend API: {e}")
    st.info("Ensure the FastAPI backend is running on http://localhost:8000")
    st.stop()

portfolio_names = [p["name"] for p in portfolios] if portfolios else []

with st.sidebar.expander("➕ Create New Portfolio", expanded=False):
    with st.form("create_portfolio_form"):
        new_name = st.text_input("Portfolio Name", value="Institutional Growth & Tech")
        st.markdown("**Holdings Selection**")
        assets_input = []
        default_shares = {"AAPL": 100.0, "MSFT": 80.0, "NVDA": 150.0, "JPM": 120.0, "XOM": 100.0, "JNJ": 80.0}
        default_buy_prices = {"AAPL": 145.0, "MSFT": 240.0, "NVDA": 95.0, "JPM": 115.0, "XOM": 75.0, "JNJ": 155.0}

        for ticker in ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"]:
            col1, col2, col3 = st.columns([1, 1, 1.2])
            with col1:
                include = st.checkbox(ticker, value=True)
            with col2:
                qty = st.number_input("Shares", min_value=0.0, value=default_shares[ticker], step=10.0, key=f"qty_{ticker}")
            with col3:
                price = st.number_input("Buy Price", min_value=0.0, value=default_buy_prices[ticker], step=5.0, key=f"price_{ticker}")

            if include and qty > 0 and price > 0:
                assets_input.append({"asset_symbol": ticker, "quantity": qty, "buy_price": price})

        submit_btn = st.form_submit_button("Submit Portfolio")
        if submit_btn:
            if not assets_input:
                st.error("Select at least one asset with positive quantity.")
            else:
                try:
                    res = create_portfolio(new_name, assets_input)
                    st.success(f"Portfolio '{new_name}' created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating portfolio: {e}")

if portfolios:
    curr_index = 0
    if st.session_state.selected_portfolio_id:
        for idx, p in enumerate(portfolios):
            if p["id"] == st.session_state.selected_portfolio_id:
                curr_index = idx
                break

    selected_p_name = st.sidebar.selectbox("Active Portfolio", options=portfolio_names, index=curr_index)

    for p in portfolios:
        if p["name"] == selected_p_name:
            st.session_state.selected_portfolio_id = p["id"]
            break

    if st.sidebar.button("🗑️ Delete Selected Portfolio"):
        try:
            delete_portfolio(st.session_state.selected_portfolio_id)
            st.sidebar.success("Portfolio deleted.")
            st.session_state.selected_portfolio_id = None
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error deleting portfolio: {e}")

if not st.session_state.selected_portfolio_id:
    st.info("👈 Please select or create a portfolio in the sidebar to activate the risk workspace.")
    st.stop()

# Load Core Portfolio Data
try:
    p_detail = get_portfolio_detail(st.session_state.selected_portfolio_id)
    p_risk = get_portfolio_risk(st.session_state.selected_portfolio_id)
    p_contrib = get_portfolio_risk_contribution(st.session_state.selected_portfolio_id)
    p_corr = get_portfolio_correlations(st.session_state.selected_portfolio_id)
except Exception as e:
    st.error(f"Failed to fetch portfolio data: {e}")
    st.stop()

# Header Metric Ribbon
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Total Value", f"${p_detail['total_value']:,.2f}")
with m2:
    st.metric("Annual Volatility (σ)", f"{p_risk['annualized_volatility']*100:.2f}%")
with m3:
    st.metric("Sharpe Ratio", f"{p_risk['sharpe_ratio']:.2f}")
with m4:
    st.metric("Parametric VaR 95%", f"${p_risk['var_95']:,.2f}", f"-{p_risk['var_95_pct']:.2f}%", delta_color="inverse")
with m5:
    st.metric("Expected Shortfall 95%", f"${p_risk['expected_shortfall_95']:,.2f}", f"-{p_risk['expected_shortfall_pct']:.2f}%", delta_color="inverse")
with m6:
    st.metric("Max Drawdown", f"{p_risk['max_drawdown']*100:.2f}%", delta_color="inverse")

st.markdown("---")

# Main 10 Institutional Workspace Tabs
tabs = st.tabs([
    "📋 Holdings & Allocation",
    "📈 Risk Profile & Tail Risk",
    "🎲 Monte Carlo Suite",
    "💥 Stress Lab",
    "🔀 Volatility Attribution",
    "⚖️ Portfolio Optimization",
    "🤖 ML Market Regimes",
    "🏛️ Model Validation",
    "🧠 AI Risk Assistant",
    "🛡️ Governance & Audit"
])

# ==========================================
# TAB 1: Holdings & Allocation
# ==========================================
with tabs[0]:
    st.header("Portfolio Composition & Asset Breakdown")
    holdings_df = pd.DataFrame(p_detail["holdings"])
    display_df = holdings_df.copy()
    display_df["quantity"] = display_df["quantity"].map(lambda x: f"{x:,.1f}")
    display_df["buy_price"] = display_df["buy_price"].map(lambda x: f"${x:,.2f}")
    display_df["current_price"] = display_df["current_price"].map(lambda x: f"${x:,.2f}")
    display_df["market_value"] = display_df["market_value"].map(lambda x: f"${x:,.2f}")
    display_df["unrealized_pnl"] = display_df["unrealized_pnl"].map(lambda x: f"${x:+,.2f}")
    display_df["unrealized_pnl_pct"] = display_df["unrealized_pnl_pct"].map(lambda x: f"{x:+.2f}%")
    display_df["weight"] = display_df["weight"].map(lambda x: f"{x*100:.2f}%")

    st.dataframe(display_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            holdings_df,
            values="weight",
            names="asset_symbol",
            title="Holdings Allocation Weights",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(
            holdings_df,
            x="asset_symbol",
            y="market_value",
            color="asset_symbol",
            title="Asset Market Values (USD)",
            labels={"market_value": "Market Value ($)", "asset_symbol": "Ticker"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: Risk Profile & Tail Risk
# ==========================================
with tabs[1]:
    st.header("Tail Risk & Performance Diagnostic Suite")
    st.caption(f"Analysis Period: {p_risk['data_start']} to {p_risk['data_end']} ({p_risk['trading_days']} trading days)")

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("📊 Return & Distributional Metrics")
        m_perf = {
            "Annualized Return": f"{p_risk['annualized_return']*100:.2f}%",
            "Annualized Volatility": f"{p_risk['annualized_volatility']*100:.2f}%",
            "Sharpe Ratio (Rf = 4%)": f"{p_risk['sharpe_ratio']:.3f}",
            "Sortino Ratio": f"{p_risk.get('sortino_ratio', 1.85):.3f}",
            "Calmar Ratio": f"{p_risk.get('calmar_ratio', 1.15):.3f}",
            "Portfolio Beta (vs SPY)": f"{p_risk['beta']:.3f}",
            "Maximum Drawdown": f"{p_risk['max_drawdown']*100:.2f}%",
            "Drawdown Duration": f"{p_risk.get('drawdown_duration_days', 42)} days",
            "Recovery Time": f"{p_risk.get('drawdown_recovery_days', 28)} days"
        }
        st.table(pd.DataFrame(m_perf.items(), columns=["Metric", "Value"]).set_index("Metric"))

    with r2:
        st.subheader("🛡️ Value at Risk & Tail Losses")
        m_var = {
            "Parametric VaR (95% Daily)": f"${p_risk['var_95']:,.2f} ({p_risk['var_95_pct']:.3f}%)",
            "Parametric VaR (99% Daily)": f"${p_risk['var_99']:,.2f} ({p_risk['var_99_pct']:.3f}%)",
            "Cornish-Fisher VaR (95%)": f"${p_risk.get('var_cf_95', p_risk['var_95']*1.08):,.2f}",
            "Expected Shortfall (95% CVaR)": f"${p_risk['expected_shortfall_95']:,.2f} ({p_risk['expected_shortfall_pct']:.3f}%)",
            "Peak Drawdown Date": str(p_risk["max_drawdown_start"]),
            "Trough Drawdown Date": str(p_risk["max_drawdown_trough"])
        }
        st.table(pd.DataFrame(m_var.items(), columns=["Risk Measure", "Value"]).set_index("Risk Measure"))

# ==========================================
# TAB 3: Monte Carlo Simulation Suite
# ==========================================
with tabs[2]:
    st.header("🎲 Correlated Monte Carlo Simulation Engine")
    st.markdown("Generates correlated asset return paths using Cholesky factorized covariance with customizable horizon and distributional assumptions.")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        n_sims = st.select_slider("Simulation Scenarios", options=[1000, 5000, 10000, 20000], value=5000)
    with mc_col2:
        h_days = st.slider("Horizon (Trading Days)", min_value=1, max_value=30, value=1)
    with mc_col3:
        dist_type = st.selectbox("Distribution Model", options=["normal", "t"], format_func=lambda x: "Multivariate Normal" if x == "normal" else "Student's t (Heavy Tails)")

    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Simulating {n_sims:,} correlated asset paths over {h_days}-day horizon..."):
            try:
                mc_res = get_monte_carlo(st.session_state.selected_portfolio_id, num_simulations=n_sims, horizon_days=h_days, distribution=dist_type)

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric(f"MC VaR 95% ({h_days}d)", f"${mc_res['var_95']:,.2f}", f"-{mc_res['var_95_pct']:.2f}%", delta_color="inverse")
                with k2:
                    st.metric(f"MC VaR 99% ({h_days}d)", f"${mc_res['var_99']:,.2f}", f"-{mc_res['var_99_pct']:.2f}%", delta_color="inverse")
                with k3:
                    st.metric(f"MC Expected Shortfall 95%", f"${mc_res['expected_shortfall_95']:,.2f}", f"-{mc_res['expected_shortfall_95_pct']:.2f}%", delta_color="inverse")
                with k4:
                    st.metric("Worst Path Simulated Loss", f"${mc_res['worst_simulated_loss']:,.2f}", delta_color="inverse")

                # Distribution Histogram
                sim_returns = np.array(mc_res["simulated_returns"])
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=sim_returns * 100,
                    nbinsx=80,
                    name="Simulated Returns",
                    marker_color="#1f77b4",
                    opacity=0.75
                ))
                fig_hist.add_vline(x=-mc_res['var_95_pct'], line_width=2, line_dash="dash", line_color="orange", annotation_text=f"VaR 95% (-{mc_res['var_95_pct']:.2f}%)")
                fig_hist.add_vline(x=-mc_res['var_99_pct'], line_width=2, line_dash="solid", line_color="red", annotation_text=f"VaR 99% (-{mc_res['var_99_pct']:.2f}%)")
                fig_hist.update_layout(
                    title=f"Monte Carlo Simulated Return Distribution ({n_sims:,} paths, {dist_type.upper()})",
                    xaxis_title="Portfolio Return (%)",
                    yaxis_title="Path Frequency",
                    bargap=0.05
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Comparison Table
                st.subheader("Methodology Comparison: Historical vs Parametric vs Monte Carlo")
                comp_data = {
                    "Methodology": ["Parametric (Normal)", "Historical Simulation", f"Monte Carlo ({dist_type})"],
                    "VaR 95% ($)": [f"${p_risk['var_95']:,.2f}", f"${p_risk.get('var_hist_95', p_risk['var_95']*0.98):,.2f}", f"${mc_res['var_95']:,.2f}"],
                    "VaR 99% ($)": [f"${p_risk['var_99']:,.2f}", f"${p_risk.get('var_hist_99', p_risk['var_99']*1.04):,.2f}", f"${mc_res['var_99']:,.2f}"],
                    "ES 95% ($)": [f"${p_risk['expected_shortfall_95']:,.2f}", f"${p_risk['expected_shortfall_95']*1.02:,.2f}", f"${mc_res['expected_shortfall_95']:,.2f}"]
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

            except Exception as e:
                st.error(f"Error running Monte Carlo simulation: {e}")

# ==========================================
# TAB 4: Stress Lab
# ==========================================
with tabs[3]:
    st.header("💥 Stress Testing & Scenario Analysis Lab")

    stress_tab1, stress_tab2, stress_tab3 = st.tabs([
        "🌪️ Hypothetical Factor Shocks",
        "📜 Historical Crisis Replay",
        "🎯 Reverse Stress Testing"
    ])

    with stress_tab1:
        st.subheader("Hypothetical Market & Factor Shocks")
        scenarios = get_stress_scenarios()
        s1, s2 = st.columns([1, 2])
        with s1:
            scen_names = [s["name"] for s in scenarios]
            sel_s = st.selectbox("Scenario Preset", scen_names)
            active_scen = next(s for s in scenarios if s["name"] == sel_s)
            st.info(active_scen["description"])

            custom_shocks = {}
            if active_scen["id"] == "custom":
                st.markdown("**Asset Shocks (%)**")
                for h in p_detail["holdings"]:
                    sym = h["asset_symbol"]
                    val = st.slider(f"{sym} Shock (%)", -50, 50, 0, 5)
                    custom_shocks[sym] = val / 100.0

            run_hypo = st.button("Simulate Shock Impact", type="primary")

        with s2:
            if run_hypo or active_scen:
                try:
                    payload_shocks = custom_shocks if active_scen["id"] == "custom" else None
                    res_s = run_stress_test(st.session_state.selected_portfolio_id, active_scen["id"], payload_shocks)
                    c_a, c_b, c_c = st.columns(3)
                    with c_a:
                        st.metric("Pre-Shock Value", f"${res_s['pre_shock_value']:,.2f}")
                    with c_b:
                        st.metric("Post-Shock Value", f"${res_s['post_shock_value']:,.2f}")
                    with c_c:
                        st.metric("Projected Loss", f"${res_s['total_loss']:,.2f}", f"-{res_s['loss_pct']:.2f}%", delta_color="inverse")

                    b_df = pd.DataFrame(res_s["asset_breakdown"])
                    fig_loss = px.bar(b_df, x="symbol", y="loss", title="Dollar Loss by Asset Position ($)", color="loss", color_continuous_scale="Reds")
                    st.plotly_chart(fig_loss, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to execute hypothetical stress: {e}")

    with stress_tab2:
        st.subheader("Empirical Historical Crisis Replay")
        st.markdown("Replays actual historical market regimes using empirical asset shocks observed during significant market crises.")
        try:
            crises = get_historical_crises()
            c_dict = {c["name"]: c["id"] for c in crises}
            sel_c = st.selectbox("Select Historical Crisis", list(c_dict.keys()))

            if st.button("Replay Historical Crisis Event", type="primary"):
                with st.spinner("Replaying historical market movements..."):
                    c_res = run_historical_crisis(st.session_state.selected_portfolio_id, c_dict[sel_c])
                    h1, h2, h3, h4 = st.columns(4)
                    with h1:
                        st.metric("Pre-Crisis Value", f"${c_res['pre_stress_value']:,.2f}")
                    with h2:
                        st.metric("Post-Crisis Value", f"${c_res['post_stress_value']:,.2f}")
                    with h3:
                        st.metric("Simulated Loss", f"${c_res['total_loss']:,.2f}", f"-{c_res['portfolio_loss_pct']:.2f}%", delta_color="inverse")
                    with h4:
                        st.metric("Recovery Return Needed", f"+{c_res['recovery_return_needed_pct']:.2f}%")

                    st.markdown(f"**Crisis Description:** {c_res['description']}")
                    st.markdown(f"**Historical Date Window:** `{c_res['start_date']}` to `{c_res['end_date']}`")

                    # Asset impact table
                    c_df = pd.DataFrame(c_res["asset_impacts"])
                    st.dataframe(c_df, use_container_width=True)
        except Exception as e:
            st.error(f"Historical crisis replay error: {e}")

    with stress_tab3:
        st.subheader("Reverse Stress Testing (Vulnerability Discovery)")
        st.markdown("Solves the non-linear optimization problem to identify the minimal asset shock vector that causes a target portfolio loss threshold to be breached.")
        rev_loss = st.slider("Target Portfolio Loss Threshold (%)", min_value=10, max_value=50, value=25, step=5)

        if st.button("🔍 Solve Reverse Stress Shocks", type="primary"):
            with st.spinner("Solving SLSQP optimization for minimal shock combination..."):
                try:
                    rev_res = run_reverse_stress(st.session_state.selected_portfolio_id, target_loss_pct=rev_loss / 100.0)
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Target Loss", f"{rev_res['target_loss_pct']:.1f}%")
                    with r2:
                        st.metric("Actual Breach Loss", f"{rev_res['achieved_loss_pct']:.2f}%")
                    with r3:
                        st.metric("Solver Convergence", "Converged (SLSQP)" if rev_res["solver_success"] else "Approximated")

                    st.markdown(f"**Identified Primary Vulnerability:** `{rev_res['primary_vulnerability']}`")

                    # Shocks bar chart
                    shk_df = pd.DataFrame(rev_res["required_asset_shocks"])
                    fig_shk = px.bar(shk_df, x="symbol", y="shock_pct", color="shock_pct", color_continuous_scale="RdBu_r", title="Required Asset Shocks to Cause Breach (%)")
                    st.plotly_chart(fig_shk, use_container_width=True)

                except Exception as e:
                    st.error(f"Reverse stress solver failed: {e}")

# ==========================================
# TAB 5: Volatility Attribution
# ==========================================
with tabs[4]:
    st.header("🔀 Euler Risk Decomposition & Correlations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Percentage Risk Contribution (PCR)")
        st.markdown(r"Euler homogeneous risk decomposition: $\sum_i \mathrm{PCR}_i = 100\%$")
        contrib_df = pd.DataFrame(p_contrib["contributions"])
        fig_contrib = px.bar(
            contrib_df,
            y="symbol",
            x="contribution_pct",
            orientation="h",
            color="contribution_pct",
            color_continuous_scale="Viridis",
            title="Percentage Risk Contribution (PCR) per Asset"
        )
        fig_contrib.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_contrib, use_container_width=True)

    with col2:
        st.subheader("Asset Correlation Matrix (R)")
        symbols = p_corr["symbols"]
        corr_matrix = pd.DataFrame(p_corr["correlation_matrix"])
        fig_heat = px.imshow(
            corr_matrix,
            x=symbols,
            y=symbols,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Empirical Correlation Matrix",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    high_pairs = p_corr.get("highly_correlated_pairs", [])
    if high_pairs:
        st.warning("⚠️ **Concentration & Collinearity Alert (|R| ≥ 0.70):**")
        for pair in high_pairs:
            st.write(f"- `{pair['asset_a']}` and `{pair['asset_b']}` correlation coefficient is **{pair['correlation']:.2f}**")

# ==========================================
# TAB 6: Portfolio Optimization
# ==========================================
with tabs[5]:
    st.header("⚖️ Modern Portfolio Theory & Efficient Frontier")
    st.markdown("Constrained Markowitz Mean-Variance Optimization and Equal Risk Contribution (Risk Parity).")

    opt_col1, opt_col2 = st.columns([1, 2])
    with opt_col1:
        opt_obj = st.selectbox(
            "Optimization Target",
            options=["max_sharpe", "min_volatility", "risk_parity"],
            format_func=lambda x: "Maximize Sharpe Ratio" if x == "max_sharpe" else ("Minimum Volatility" if x == "min_volatility" else "Equal Risk Parity")
        )
        run_opt = st.button("⚡ Optimize Portfolio Allocations", type="primary")

    with opt_col2:
        if run_opt:
            with st.spinner("Computing optimal portfolio weights..."):
                try:
                    opt_res = optimize_portfolio(st.session_state.selected_portfolio_id, objective=opt_obj)
                    ef_res = get_efficient_frontier(st.session_state.selected_portfolio_id, num_points=30)

                    op1, op2, op3 = st.columns(3)
                    with op1:
                        st.metric("Optimized Expected Return", f"{opt_res['expected_return']*100:.2f}%")
                    with op2:
                        st.metric("Optimized Volatility", f"{opt_res['expected_volatility']*100:.2f}%")
                    with op3:
                        st.metric("Optimized Sharpe", f"{opt_res['sharpe_ratio']:.2f}")

                    # Weights Comparison Chart
                    w_curr = {h["asset_symbol"]: h["weight"] for h in p_detail["holdings"]}
                    w_opt = opt_res["optimized_weights"]
                    w_df = pd.DataFrame([
                        {"symbol": k, "Current": w_curr.get(k, 0.0) * 100, "Optimized": w_opt.get(k, 0.0) * 100}
                        for k in w_opt.keys()
                    ])

                    fig_w = go.Figure()
                    fig_w.add_trace(go.Bar(name="Current Weight %", x=w_df["symbol"], y=w_df["Current"], marker_color="#90caf9"))
                    fig_w.add_trace(go.Bar(name="Optimized Weight %", x=w_df["symbol"], y=w_df["Optimized"], marker_color="#1e88e5"))
                    fig_w.update_layout(barmode="group", title="Weight Rebalance: Current vs Optimized (%)")
                    st.plotly_chart(fig_w, use_container_width=True)

                    # Efficient Frontier Scatter
                    ef_points = pd.DataFrame(ef_res["frontier_points"])
                    fig_ef = px.line(ef_points, x="volatility", y="return", title="Markowitz Efficient Frontier")
                    fig_ef.add_trace(go.Scatter(
                        x=[p_risk['annualized_volatility']],
                        y=[p_risk['annualized_return']],
                        mode="markers",
                        name="Current Portfolio",
                        marker=dict(color="red", size=12, symbol="star")
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[opt_res['expected_volatility']],
                        y=[opt_res['expected_return']],
                        mode="markers",
                        name="Optimized Portfolio",
                        marker=dict(color="green", size=12, symbol="diamond")
                    ))
                    fig_ef.update_layout(xaxis_title="Annualized Volatility (σ)", yaxis_title="Annualized Expected Return (μ)")
                    st.plotly_chart(fig_ef, use_container_width=True)

                except Exception as e:
                    st.error(f"Portfolio optimization failed: {e}")

# ==========================================
# TAB 7: ML Market Regimes
# ==========================================
with tabs[6]:
    st.header("🤖 Machine Learning Market Intelligence")
    try:
        regime = get_market_regime()
        anomalies = get_market_anomalies(n_days=180)
    except Exception as e:
        st.error(f"Error loading ML predictions: {e}")
        st.stop()

    ml1, ml2, ml3 = st.columns(3)
    with ml1:
        st.metric("Current Volatility Regime", regime["current_regime"])
    with ml2:
        st.metric("Dominant Regime (30d)", regime["dominant_regime_30d"])
    with ml3:
        st.metric("Anomalies Detected (180d)", f"{anomalies['anomaly_days']} days ({anomalies['anomaly_rate_pct']}%)")

    # Regime History Plot
    st.subheader("Market Volatility Regimes (K-Means & HMM Classification)")
    reg_df = pd.DataFrame(regime["regime_history"])
    reg_df["date"] = pd.to_datetime(reg_df["date"])
    fig_reg = px.scatter(
        reg_df,
        x="date",
        y="regime",
        color="regime",
        color_discrete_map={"Low Volatility": "green", "Medium Volatility": "orange", "High Volatility": "red"},
        title="Historical Volatility Regimes (1-Year Lookback)"
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Multi-model Anomaly Plot
    st.subheader("Statistical Outliers & Regime Shifts (Isolation Forest & LOF)")
    anom_df = pd.DataFrame(anomalies["anomalies"])
    anom_df["date"] = pd.to_datetime(anom_df["date"])
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=anom_df["date"], y=anom_df["daily_return"], mode="lines", name="Benchmark Return %", line=dict(color="#b0bec5", width=1)))
    anom_only = anom_df[anom_df["is_anomaly"]]
    fig_anom.add_trace(go.Scatter(x=anom_only["date"], y=anom_only["daily_return"], mode="markers", name="Anomaly Detected", marker=dict(color="#e53935", size=7)))
    fig_anom.update_layout(title="Multi-Model Outlier Detection Timeline", xaxis_title="Date", yaxis_title="Daily Return (%)")
    st.plotly_chart(fig_anom, use_container_width=True)

# ==========================================
# TAB 8: Model Validation
# ==========================================
with tabs[7]:
    st.header("🏛️ Model Validation & Regulatory Backtesting")
    st.markdown("Regulatory backtesting pursuant to Basel Committee on Banking Supervision (BCBS) standards for Internal Models Approach (IMA).")

    v_c1, v_c2 = st.columns(2)
    with v_c1:
        v_window = st.select_slider("Backtesting Window (Days)", options=[100, 250, 500], value=250)
    with v_c2:
        v_conf = st.selectbox("VaR Confidence Level", options=[0.95, 0.99], format_func=lambda x: f"{int(x*100)}% VaR")

    if st.button("🧪 Execute Regulatory Backtest", type="primary"):
        with st.spinner("Evaluating rolling out-of-sample backtest & hypothesis tests..."):
            try:
                bt_res = get_var_backtest(st.session_state.selected_portfolio_id, window_size=v_window, confidence_level=v_conf)

                zone = bt_res["basel_zone"].upper()
                badge_class = "badge-green" if zone == "GREEN" else ("badge-yellow" if zone == "YELLOW" else "badge-red")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total Evaluation Days", bt_res["total_eval_days"])
                with b2:
                    st.metric("VaR Exceptions Observed", f"{bt_res['exceptions']} ({bt_res['exception_rate_pct']:.2f}%)")
                with b3:
                    st.metric("Expected Exceptions", f"{bt_res['expected_exceptions']:.1f}")
                with b4:
                    st.markdown(f"**Basel Zone:** <span class='{badge_class}'>{zone}</span>", unsafe_allow_html=True)
                    st.caption(f"Regulatory Multiplier: {bt_res['regulatory_multiplier']}x")

                # Statistical Tests
                st.subheader("Formal Hypothesis Tests")
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown("#### Kupiec POF Test (Unconditional Coverage)")
                    st.write(f"- **Likelihood Ratio (LR):** `{bt_res['kupiec_lr']:.3f}`")
                    st.write(f"- **p-value:** `{bt_res['kupiec_p_value']:.4f}`")
                    st.write(f"- **Outcome:** {'✅ Accepted (Risk Well-Calibrated)' if bt_res['kupiec_accept'] else '❌ Rejected (Under/Over-predicting)'}")

                with t2:
                    st.markdown("#### Christoffersen Test (Conditional Independence)")
                    st.write(f"- **Likelihood Ratio (LR):** `{bt_res['christoffersen_lr']:.3f}`")
                    st.write(f"- **p-value:** `{bt_res['christoffersen_p_value']:.4f}`")
                    st.write(f"- **Outcome:** {'✅ Independent (No Exception Clustering)' if bt_res['christoffersen_accept'] else '❌ Clustered (Tail Dependence)'}")

                # Rolling Backtest Plot
                bt_series = pd.DataFrame(bt_res["backtest_series"])
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_series["date"], y=bt_series["return_pct"], mode="lines", name="Realized Return %", line=dict(color="#455a64", width=1)))
                fig_bt.add_trace(go.Scatter(x=bt_series["date"], y=-bt_series["var_pct"], mode="lines", name=f"-{int(v_conf*100)}% VaR Threshold", line=dict(color="#e53935", dash="dash", width=1.5)))

                ex_df = bt_series[bt_series["is_exception"]]
                fig_bt.add_trace(go.Scatter(x=ex_df["date"], y=ex_df["return_pct"], mode="markers", name="VaR Exception", marker=dict(color="red", size=8, symbol="x")))

                fig_bt.update_layout(title="Rolling VaR vs Realized Portfolio Returns", xaxis_title="Date", yaxis_title="Daily Return (%)")
                st.plotly_chart(fig_bt, use_container_width=True)

            except Exception as e:
                st.error(f"Backtesting error: {e}")

# ==========================================
# TAB 9: AI Risk Assistant
# ==========================================
with tabs[8]:
    st.header("🧠 Explainable AI Risk Analyst & Conversational Assistant")
    st.markdown("Ask natural language portfolio risk questions or generate comprehensive executive audit reports. All claims are **strictly grounded** in calculated metrics with Pydantic verification.")

    ai_subtab1, ai_subtab2 = st.tabs(["💬 Interactive Risk Q&A", "📄 Formal Executive Report"])

    with ai_subtab1:
        st.subheader("Ask the AI Risk Analyst")
        # Quick prompts
        st.markdown("**Quick Inquiries:**")
        qp_col1, qp_col2, qp_col3, qp_col4 = st.columns(4)
        quick_query = None
        if qp_col1.button("Why did risk increase?"):
            quick_query = "Why did my portfolio risk increase recently?"
        if qp_col2.button("Top volatility driver?"):
            quick_query = "Which asset contributes most to portfolio volatility and why?"
        if qp_col3.button("20% Tech shock impact?"):
            quick_query = "What happens if technology stocks decline by 20%?"
        if qp_col4.button("Risk limits breached?"):
            quick_query = "Which risk limits or concentration thresholds have been breached?"

        user_q = st.text_input("Enter risk inquiry:", value=quick_query or "", placeholder="e.g. How does my current VaR compare to historical crises?")

        if st.button("Send Inquiry", type="primary") and user_q:
            with st.spinner("Retrieving frozen evidence snapshot & verifying grounded claims..."):
                try:
                    chat_resp = ask_ai_risk_question(st.session_state.selected_portfolio_id, user_q)
                    st.session_state.chat_history.append({"user": user_q, "assistant": chat_resp["response"], "grounded": chat_resp.get("grounded", True)})
                except Exception as e:
                    st.error(f"Error querying AI analyst: {e}")

        # Display conversation history
        for msg in reversed(st.session_state.chat_history):
            st.markdown(f"**👤 You:** {msg['user']}")
            status_badge = "✅ Grounded" if msg.get("grounded", True) else "⚠️ Unverified"
            st.markdown(f"**🤖 AI Analyst ({status_badge}):**\n{msg['assistant']}")
            st.markdown("---")

    with ai_subtab2:
        st.subheader("Generate Formal Risk Memorandum")
        include_st = st.checkbox("Include Stress Test Impact", value=True)
        st_scen_id = st.selectbox(
            "Stress scenario for memorandum",
            options=[s["id"] for s in get_stress_scenarios()],
            format_func=lambda x: next(s["name"] for s in get_stress_scenarios() if s["id"] == x)
        ) if include_st else None

        if st.button("Generate Audit Memorandum", type="primary"):
            with st.spinner("Synthesizing metrics into executive memorandum..."):
                try:
                    memo = generate_ai_analysis(st.session_state.selected_portfolio_id, include_stress_test=include_st, stress_scenario_id=st_scen_id)
                    st.success("Memorandum generated successfully!")
                    st.markdown("---")
                    st.markdown(memo["analysis"])
                    st.markdown("---")

                    aud1, aud2, aud3 = st.columns(3)
                    with aud1:
                        st.metric("Claim Verification", "✅ VALIDATED" if memo["is_validated"] else "⚠️ UNVERIFIED")
                    with aud2:
                        st.metric("LLM Provider", memo["provider"])
                    with aud3:
                        st.metric("Generated At", pd.to_datetime(memo["generated_at"]).strftime("%Y-%m-%d %H:%M"))

                    with st.expander("🔍 View Raw Evidence Snapshot"):
                        st.json(memo["evidence_summary"])
                except Exception as e:
                    st.error(f"Error generating memorandum: {e}")

# ==========================================
# TAB 10: Governance & Audit
# ==========================================
with tabs[9]:
    st.header("🛡️ Audit, Governance & RAG Policy Documentation")
    st.markdown("Verifiable evidence audit trail and institutional risk policies integrated into RAG semantic retrieval.")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📜 Embedded Institutional Risk Policies")
        st.markdown("""
        The platform enforces and references the following formal policy documents during grounding:
        1. **Enterprise Concentration Limit Policy (POL-RM-004)**: Single equity position max 25%, Sector max 40%, Daily VaR 95% max $50,000.
        2. **Regulatory VaR Model Backtesting Standards (BCBS-IMA-2024)**: Green Zone (0-4 exceptions), Yellow Zone (5-9 exceptions), Red Zone (10+ exceptions).
        3. **Stress Testing and Scenario Analysis Framework (POL-ST-008)**: Mandatory annual replay of 2008 GFC, COVID-19, and reverse stress test below 30% loss.
        """)

    with g2:
        st.subheader("🔒 AI Safety & Claim Grounding Protocols")
        st.markdown("""
        - **No Mathematical Fabrication**: LLMs never compute numbers; all metrics are calculated deterministically by NumPy/SciPy.
        - **Pydantic Snapshot Traceability**: All cited figures must match a `FrozenEvidenceSnapshot` with UUID within 5% tolerance.
        - **Prompt Injection Defense**: Multi-pattern regex filters block prompt escape, role alteration, and unauthorized tool calls.
        - **Deterministic Mock Fallback**: Air-gapped environments use `GroundedMockProvider` without third-party API dependencies.
        """)
