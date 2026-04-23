import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Miller Construction – Contract Bidding Simulator",
    page_icon="🏗️",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1a3e5c;
        text-align: center; padding: 0.5rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #5a7d9a; text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f0fe 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #d0dce8; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a3e5c; }
    .metric-label { font-size: 0.85rem; color: #5a7d9a; margin-top: 0.2rem; }
    .positive { color: #0e8a16; }
    .negative { color: #d73a49; }
    .section-divider { border: none; border-top: 2px solid #e0e8f0; margin: 1.5rem 0; }
    div[data-testid="stSidebar"] { background: #f4f8fc; }
    .fit-best { background: #d4edda; border-radius: 6px; padding: 0.3rem 0.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏗️ Miller Construction – Contract Bidding Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Monte Carlo simulation to decide whether to bid and at what price</div>', unsafe_allow_html=True)


# ─── Distribution Fitting Engine ──────────────────────────────────────────────
CANDIDATE_DISTS = {
    "Normal": stats.norm,
    "Log-Normal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Uniform": stats.uniform,
    "Triangular": stats.triang,
    "Beta": stats.beta,
    "Logistic": stats.logistic,
}

# Distributions whose support starts at zero — pin loc=0 during fitting
# (matches reference implementation best practice)
ZERO_LOWER_BOUND_DISTS = {"Exponential", "Gamma", "Log-Normal", "Weibull"}

COMPETITOR_DIST_OPTIONS = ["Triangular", "Normal", "Uniform", "Log-Normal"]


def compute_chi_square_gof(dist, params, data, max_bins=10):
    """Chi-square goodness-of-fit test with bin merging (expected >= 5)."""
    n = len(data)
    quantiles = np.linspace(0, 1, max_bins + 1)
    bin_edges = dist.ppf(quantiles, *params)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    distinct = [bin_edges[0]]
    for e in bin_edges[1:]:
        if e > distinct[-1]:
            distinct.append(e)
    if len(distinct) < 3:
        return np.nan, np.nan

    observed, _ = np.histogram(data, bins=np.array(distinct))
    expected_per_bin = n / (len(distinct) - 1)

    merged_obs, merged_exp = [], []
    run_obs, run_exp = 0.0, 0.0
    for obs in observed:
        run_obs += float(obs)
        run_exp += expected_per_bin
        if run_exp >= 5:
            merged_obs.append(run_obs)
            merged_exp.append(run_exp)
            run_obs, run_exp = 0.0, 0.0
    if run_exp > 0:
        if merged_exp:
            merged_obs[-1] += run_obs
            merged_exp[-1] += run_exp
        else:
            merged_obs.append(run_obs)
            merged_exp.append(run_exp)

    dof = len(merged_obs) - 1 - len(params)
    if dof <= 0:
        return np.nan, np.nan

    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(merged_obs, merged_exp))
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    return chi2_stat, p_value


def fit_all_distributions(data):
    """Fit candidate distributions and return sorted results by BIC."""
    results = []
    n = len(data)
    for name, dist in CANDIDATE_DISTS.items():
        try:
            # Pin loc=0 for distributions with natural zero lower bound
            if name in ZERO_LOWER_BOUND_DISTS:
                params = dist.fit(data, floc=0)
            else:
                params = dist.fit(data)

            log_lik_vals = dist.logpdf(data, *params)
            if not np.all(np.isfinite(log_lik_vals)):
                continue
            log_lik = float(np.sum(log_lik_vals))

            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik
            ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
            _, chi2_p = compute_chi_square_gof(dist, params, data)

            results.append({
                "Distribution": name,
                "Parameters": params,
                "Log-Likelihood": log_lik,
                "AIC": aic,
                "BIC": bic,
                "KS Statistic": ks_stat,
                "KS p-value": ks_p,
                "Chi-sq p-value": chi2_p,
                "scipy_dist": dist,
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["BIC"])
    return results


def sample_from_fit(dist_obj, params, size, rng, floor=None):
    """Draw random samples from a fitted scipy distribution."""
    samples = dist_obj.rvs(*params, size=size, random_state=rng)
    if floor is not None:
        samples = np.maximum(samples, floor)
    return samples


def sample_competitor(dist_type, params_dict, size, rng):
    """Sample competitor bids from the chosen distribution type."""
    if dist_type == "Triangular":
        return rng.triangular(params_dict["min"], params_dict["mode"], params_dict["max"], size)
    elif dist_type == "Normal":
        mu = params_dict["mode"]
        sigma = (params_dict["max"] - params_dict["min"]) / 4
        samples = rng.normal(mu, sigma, size)
        return np.clip(samples, params_dict["min"], params_dict["max"])
    elif dist_type == "Uniform":
        return rng.uniform(params_dict["min"], params_dict["max"], size)
    elif dist_type == "Log-Normal":
        mu = np.log(params_dict["mode"])
        sigma = (np.log(params_dict["max"]) - np.log(params_dict["min"])) / 4
        samples = rng.lognormal(mu, sigma, size)
        return np.clip(samples, params_dict["min"], params_dict["max"])
    else:
        return rng.triangular(params_dict["min"], params_dict["mode"], params_dict["max"], size)


# ─── Sidebar — All Configuration ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🔢 Simulation")
    n_sims = st.slider("Number of simulations", 1000, 100000, 10000, step=1000,
                        help="More simulations = more accurate but slower")

    st.markdown("---")
    st.subheader("🏢 Competitor Settings")

    comp_dist_type = st.selectbox("Competitor bid distribution",
                                  COMPETITOR_DIST_OPTIONS, index=0,
                                  help="Distribution used to model all competitor bids")

    st.markdown("**Competitor A** *(always bids)*")
    comp_a_min = st.number_input("Min bid ($)", value=90000, step=1000, key="ca_min")
    comp_a_mode = st.number_input("Most likely bid ($)", value=130000, step=1000, key="ca_mode")
    comp_a_max = st.number_input("Max bid ($)", value=180000, step=1000, key="ca_max")

    st.markdown("---")
    st.markdown("**Competitors B & C** *(uncertain participation)*")
    prob_b = st.slider("Probability Comp B bids", 0.0, 1.0, 0.50, 0.05, key="pb")
    prob_c = st.slider("Probability Comp C bids", 0.0, 1.0, 0.50, 0.05, key="pc_prob")
    st.caption("If they bid, same distribution & parameters as Comp A.")

    st.markdown("---")
    st.subheader("📏 Constraints")
    min_project_cost = st.number_input("Min project cost floor ($)", value=70000, step=1000,
                                        help="Project completion cost can never be less than this")

comp_params = {"min": comp_a_min, "mode": comp_a_mode, "max": comp_a_max}


# ─── Data Loading ─────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

data_col1, data_col2 = st.columns([2, 1])
with data_col1:
    st.subheader("📂 Historical Data")
    data_source = st.radio(
        "Choose data source:",
        ["Use default file (project_costs.csv)", "Upload a CSV file"],
        horizontal=True,
    )

uploaded_df = None
if data_source == "Upload a CSV file":
    with data_col2:
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="CSV must have two columns: bid preparation costs and total project costs. "
                 "Rows where the project was not won should have the second column empty.",
        )
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
    else:
        st.info("⬆️ Please upload a CSV file to continue, or switch to the default file.")
        st.stop()
else:
    uploaded_df = pd.read_csv("project_costs.csv")

# Standardise column names
uploaded_df.columns = ["bid_prep_cost", "project_cost"]
bid_prep_all = uploaded_df["bid_prep_cost"].dropna().values.astype(float)
project_costs_won = uploaded_df["project_cost"].dropna().values.astype(float)
project_costs_won = np.maximum(project_costs_won, min_project_cost)

st.success(
    f"✅ Data loaded — **{len(bid_prep_all)}** bid records, "
    f"**{len(project_costs_won)}** with project costs (won bids)."
)


# ─── Distribution Fitting ─────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("📐 Best-Fit Distributions")
st.caption("Multiple candidate distributions are fitted to your data and ranked by BIC (lower is better).")

bp_fits = fit_all_distributions(bid_prep_all)
pc_fits = fit_all_distributions(project_costs_won)

best_bp = bp_fits[0]
best_pc = pc_fits[0]

fit_col1, fit_col2 = st.columns(2)

with fit_col1:
    st.markdown("#### Bid Preparation Cost")
    st.markdown(
        f'<div class="fit-best">🏆 Best fit: <b>{best_bp["Distribution"]}</b> '
        f'(BIC = {best_bp["BIC"]:,.1f}, KS p = {best_bp["KS p-value"]:.4f})</div>',
        unsafe_allow_html=True,
    )
    bp_fit_df = pd.DataFrame([{
        "Distribution": r["Distribution"],
        "BIC": f'{r["BIC"]:,.1f}',
        "AIC": f'{r["AIC"]:,.1f}',
        "KS Statistic": f'{r["KS Statistic"]:.4f}',
        "KS p-value": f'{r["KS p-value"]:.4f}',
        "Chi-sq p-value": f'{r["Chi-sq p-value"]:.4f}' if np.isfinite(r["Chi-sq p-value"]) else "N/A",
    } for r in bp_fits])
    st.dataframe(bp_fit_df, use_container_width=True, hide_index=True)

    fig_bp_fit = go.Figure()
    fig_bp_fit.add_trace(go.Histogram(
        x=bid_prep_all, nbinsx=30, histnorm="probability density",
        marker_color="#f4a261", opacity=0.7, name="Data",
    ))
    x_range = np.linspace(bid_prep_all.min(), bid_prep_all.max(), 300)
    for r in bp_fits[:3]:
        pdf_vals = r["scipy_dist"].pdf(x_range, *r["Parameters"])
        fig_bp_fit.add_trace(go.Scatter(
            x=x_range, y=pdf_vals, mode="lines", name=r["Distribution"],
            line=dict(width=2.5 if r == best_bp else 1.5,
                      dash="solid" if r == best_bp else "dot"),
        ))
    fig_bp_fit.update_layout(
        title="Bid Prep Cost — Data vs Fitted PDFs",
        xaxis_title="Cost ($)", yaxis_title="Density",
        template="plotly_white", height=350, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_bp_fit, use_container_width=True)

with fit_col2:
    st.markdown("#### Project Completion Cost")
    st.markdown(
        f'<div class="fit-best">🏆 Best fit: <b>{best_pc["Distribution"]}</b> '
        f'(BIC = {best_pc["BIC"]:,.1f}, KS p = {best_pc["KS p-value"]:.4f})</div>',
        unsafe_allow_html=True,
    )
    pc_fit_df = pd.DataFrame([{
        "Distribution": r["Distribution"],
        "BIC": f'{r["BIC"]:,.1f}',
        "AIC": f'{r["AIC"]:,.1f}',
        "KS Statistic": f'{r["KS Statistic"]:.4f}',
        "KS p-value": f'{r["KS p-value"]:.4f}',
        "Chi-sq p-value": f'{r["Chi-sq p-value"]:.4f}' if np.isfinite(r["Chi-sq p-value"]) else "N/A",
    } for r in pc_fits])
    st.dataframe(pc_fit_df, use_container_width=True, hide_index=True)

    x_range_pc = np.linspace(project_costs_won.min(), project_costs_won.max(), 300)
    fig_pc_fit = go.Figure()
    fig_pc_fit.add_trace(go.Histogram(
        x=project_costs_won, nbinsx=30, histnorm="probability density",
        marker_color="#e76f51", opacity=0.7, name="Data",
    ))
    for r in pc_fits[:3]:
        pdf_vals = r["scipy_dist"].pdf(x_range_pc, *r["Parameters"])
        fig_pc_fit.add_trace(go.Scatter(
            x=x_range_pc, y=pdf_vals, mode="lines", name=r["Distribution"],
            line=dict(width=2.5 if r == best_pc else 1.5,
                      dash="solid" if r == best_pc else "dot"),
        ))
    fig_pc_fit.update_layout(
        title="Project Cost — Data vs Fitted PDFs",
        xaxis_title="Cost ($)", yaxis_title="Density",
        template="plotly_white", height=350, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_pc_fit, use_container_width=True)

# Override distribution selection
bp_names = [r["Distribution"] for r in bp_fits]
pc_names = [r["Distribution"] for r in pc_fits]

with st.expander("🔧 Override distribution selection (optional)"):
    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        sel_bp = st.selectbox("Bid Prep Cost distribution", bp_names, index=0, key="sel_bp")
    with ov_col2:
        sel_pc = st.selectbox("Project Cost distribution", pc_names, index=0, key="sel_pc")

chosen_bp = next(r for r in bp_fits if r["Distribution"] == sel_bp)
chosen_pc = next(r for r in pc_fits if r["Distribution"] == sel_pc)

# Show active selections in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Active Distributions")
    st.info(f"**Bid Prep:** {chosen_bp['Distribution']}")
    st.info(f"**Project Cost:** {chosen_pc['Distribution']}")
    st.info(f"**Competitors:** {comp_dist_type}")


# ─── Simulation Engine ─────────────────────────────────────────────────────────
def run_simulation(n, miller_bid, comp_dist, comp_prms, p_b, p_c,
                   bp_dist, bp_params, pc_dist, pc_params, min_pc, seed=42):
    rng = np.random.default_rng(seed)

    bid_prep = sample_from_fit(bp_dist, bp_params, n, rng)
    bid_prep = np.clip(bid_prep, 0, None)

    proj_cost = sample_from_fit(pc_dist, pc_params, n, rng)
    proj_cost = np.maximum(proj_cost, min_pc)

    comp_a_bids = sample_competitor(comp_dist, comp_prms, n, rng)

    b_flag = rng.random(n) < p_b
    comp_b_bids = np.where(b_flag, sample_competitor(comp_dist, comp_prms, n, rng), np.inf)

    c_flag = rng.random(n) < p_c
    comp_c_bids = np.where(c_flag, sample_competitor(comp_dist, comp_prms, n, rng), np.inf)

    lowest_comp = np.minimum(comp_a_bids, np.minimum(comp_b_bids, comp_c_bids))
    miller_wins = miller_bid < lowest_comp

    profit = np.where(miller_wins, miller_bid - proj_cost - bid_prep, -bid_prep)

    return {
        "bid_prep": bid_prep, "proj_cost": proj_cost,
        "comp_a_bids": comp_a_bids, "comp_b_bids": comp_b_bids,
        "comp_c_bids": comp_c_bids, "b_flag": b_flag, "c_flag": c_flag,
        "lowest_comp": lowest_comp, "miller_wins": miller_wins, "profit": profit,
    }


def run_tradeoff(bid_range, n, comp_dist, comp_prms, p_b, p_c,
                 bp_dist, bp_params, pc_dist, pc_params, min_pc):
    rows = []
    for bid in bid_range:
        s = run_simulation(n, bid, comp_dist, comp_prms, p_b, p_c,
                           bp_dist, bp_params, pc_dist, pc_params, min_pc)
        rows.append({
            "bid": bid,
            "win_rate": s["miller_wins"].mean(),
            "expected_profit": s["profit"].mean(),
            "prob_positive": (s["profit"] > 0).mean(),
        })
    return pd.DataFrame(rows)


# ─── Simulation Results ───────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("🎲 Simulation Results")

# Miller's bid amount — now in the main body, not sidebar
bid_col1, bid_col2 = st.columns([1, 2])
with bid_col1:
    miller_bid = st.number_input("💰 Miller's Bid Amount ($)", value=130000,
                                  min_value=50000, max_value=500000, step=1000,
                                  format="%d",
                                  help="The dollar amount Miller will submit as their bid")

sim = run_simulation(
    n_sims, miller_bid, comp_dist_type, comp_params, prob_b, prob_c,
    chosen_bp["scipy_dist"], chosen_bp["Parameters"],
    chosen_pc["scipy_dist"], chosen_pc["Parameters"],
    min_project_cost,
)

# ─── Key Metrics ───────────────────────────────────────────────────────────────
win_rate = sim["miller_wins"].mean()
expected_profit = sim["profit"].mean()
profit_std = sim["profit"].std()
prob_positive = (sim["profit"] > 0).mean()
median_profit = np.median(sim["profit"])
should_bid = expected_profit > 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    emoji = "✅" if should_bid else "❌"
    text = "YES" if should_bid else "NO"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{emoji} {text}</div>
        <div class="metric-label">Should Miller Bid?</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">${miller_bid:,.0f}</div>
        <div class="metric-label">Miller's Bid Amount</div>
    </div>""", unsafe_allow_html=True)

with col3:
    cc = "positive" if expected_profit > 0 else "negative"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value {cc}">${expected_profit:,.0f}</div>
        <div class="metric-label">Expected Profit / Loss</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{win_rate:.1%}</div>
        <div class="metric-label">Win Probability</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{prob_positive:.1%}</div>
        <div class="metric-label">Prob. of Positive Profit</div>
    </div>""", unsafe_allow_html=True)


# ─── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Profit Distribution",
    "🔄 Tradeoff Curve",
    "📊 Input Distributions",
    "📋 Data Summary",
])

# ── Tab 1: Profit Distribution ─────────────────────────────────────────────────
with tab1:
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sim["profit"], nbinsx=80,
            marker_color="#4a90d9", opacity=0.85, name="Profit",
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", line_width=2,
                           annotation_text="Break-even", annotation_position="top right")
        fig_hist.add_vline(x=expected_profit, line_dash="dot", line_color="green", line_width=2,
                           annotation_text=f"Mean: ${expected_profit:,.0f}",
                           annotation_position="top left")
        fig_hist.update_layout(
            title="Profit / Loss Distribution",
            xaxis_title="Profit ($)", yaxis_title="Frequency",
            template="plotly_white", height=420, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with p_col2:
        sorted_p = np.sort(sim["profit"])
        cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_p, y=cdf, mode="lines",
            line=dict(color="#4a90d9", width=2), name="CDF",
        ))
        fig_cdf.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
        fig_cdf.update_layout(
            title="Cumulative Distribution of Profit",
            xaxis_title="Profit ($)", yaxis_title="Cumulative Probability",
            template="plotly_white", height=420, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

    st.markdown("#### Profit Summary Statistics")
    stats_df = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Std Dev", "5th Pctl", "25th Pctl",
                       "75th Pctl", "95th Pctl", "Min", "Max"],
        "Value ($)": [
            f"${expected_profit:,.0f}", f"${median_profit:,.0f}", f"${profit_std:,.0f}",
            f"${np.percentile(sim['profit'], 5):,.0f}",
            f"${np.percentile(sim['profit'], 25):,.0f}",
            f"${np.percentile(sim['profit'], 75):,.0f}",
            f"${np.percentile(sim['profit'], 95):,.0f}",
            f"${sim['profit'].min():,.0f}", f"${sim['profit'].max():,.0f}",
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ── Tab 2: Tradeoff Curve ─────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Bid Amount vs Expected Profit & Win Probability")
    st.caption("Explore how changing Miller's bid affects profitability and the chance of winning.")

    bid_lo, bid_hi = st.slider("Bid range to explore ($)", 80000, 250000,
                                (90000, 200000), step=5000, key="tradeoff_range")
    bid_range = np.arange(bid_lo, bid_hi + 1, 2000)

    with st.spinner("Running tradeoff simulations…"):
        tradeoff_df = run_tradeoff(
            bid_range, min(n_sims, 10000),
            comp_dist_type, comp_params, prob_b, prob_c,
            chosen_bp["scipy_dist"], chosen_bp["Parameters"],
            chosen_pc["scipy_dist"], chosen_pc["Parameters"],
            min_project_cost,
        )

    fig_trade = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trade.add_trace(
        go.Scatter(x=tradeoff_df["bid"], y=tradeoff_df["expected_profit"],
                   mode="lines+markers", name="Expected Profit ($)",
                   line=dict(color="#0e8a16", width=2.5), marker=dict(size=5)),
        secondary_y=False,
    )
    fig_trade.add_trace(
        go.Scatter(x=tradeoff_df["bid"], y=tradeoff_df["win_rate"],
                   mode="lines+markers", name="Win Probability",
                   line=dict(color="#4a90d9", width=2.5, dash="dot"), marker=dict(size=5)),
        secondary_y=True,
    )
    fig_trade.add_hline(y=0, line_dash="dash", line_color="red", line_width=1, secondary_y=False)

    if len(tradeoff_df) > 0:
        best = tradeoff_df.loc[tradeoff_df["expected_profit"].idxmax()]
        fig_trade.add_trace(
            go.Scatter(x=[best["bid"]], y=[best["expected_profit"]],
                       mode="markers+text", name="Optimal Bid",
                       marker=dict(color="gold", size=14, symbol="star",
                                   line=dict(color="black", width=1.5)),
                       text=[f"${best['bid']:,.0f}"], textposition="top center"),
            secondary_y=False,
        )

    fig_trade.update_layout(
        title="Tradeoff: Expected Profit vs Win Probability",
        xaxis_title="Miller's Bid ($)",
        template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=40),
    )
    fig_trade.update_yaxes(title_text="Expected Profit ($)", secondary_y=False)
    fig_trade.update_yaxes(title_text="Win Probability", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig_trade, use_container_width=True)

    if len(tradeoff_df) > 0:
        st.success(
            f"⭐ **Optimal bid in range:** ${best['bid']:,.0f}  —  "
            f"Expected profit: ${best['expected_profit']:,.0f}  |  "
            f"Win probability: {best['win_rate']:.1%}"
        )

    st.dataframe(
        tradeoff_df.rename(columns={
            "bid": "Bid ($)", "win_rate": "Win Probability",
            "expected_profit": "Expected Profit ($)", "prob_positive": "P(Profit > 0)",
        }).style.format({
            "Bid ($)": "${:,.0f}", "Win Probability": "{:.1%}",
            "Expected Profit ($)": "${:,.0f}", "P(Profit > 0)": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )


# ── Tab 3: Input Distributions ────────────────────────────────────────────────
with tab3:
    ix1, ix2 = st.columns(2)

    with ix1:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Histogram(x=bid_prep_all, nbinsx=30,
                                       marker_color="#f4a261", opacity=0.85, name="Historical"))
        fig_bp.add_trace(go.Histogram(x=sim["bid_prep"], nbinsx=30,
                                       marker_color="#264653", opacity=0.5, name="Simulated"))
        fig_bp.update_layout(
            title="Bid Preparation Cost", xaxis_title="Cost ($)", yaxis_title="Frequency",
            barmode="overlay", template="plotly_white", height=380, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_bp, use_container_width=True)

    with ix2:
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Histogram(x=project_costs_won, nbinsx=30,
                                       marker_color="#e76f51", opacity=0.85, name="Historical"))
        fig_pc.add_trace(go.Histogram(x=sim["proj_cost"], nbinsx=30,
                                       marker_color="#264653", opacity=0.5, name="Simulated"))
        fig_pc.update_layout(
            title="Project Completion Cost", xaxis_title="Cost ($)", yaxis_title="Frequency",
            barmode="overlay", template="plotly_white", height=380, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("#### Competitor Bid Distributions")
    cx1, cx2 = st.columns(2)

    with cx1:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(x=sim["comp_a_bids"], nbinsx=40,
                                         marker_color="#2a9d8f", opacity=0.7, name="Comp A"))
        fb = sim["comp_b_bids"][sim["b_flag"]]
        if len(fb) > 0:
            fig_comp.add_trace(go.Histogram(x=fb, nbinsx=40,
                                             marker_color="#e9c46a", opacity=0.6,
                                             name="Comp B (when bidding)"))
        fc = sim["comp_c_bids"][sim["c_flag"]]
        if len(fc) > 0:
            fig_comp.add_trace(go.Histogram(x=fc, nbinsx=40,
                                             marker_color="#e76f51", opacity=0.5,
                                             name="Comp C (when bidding)"))
        fig_comp.add_vline(x=miller_bid, line_dash="dash", line_color="blue", line_width=2,
                           annotation_text=f"Miller: ${miller_bid:,.0f}")
        fig_comp.update_layout(
            title=f"Competitor Bid Amounts ({comp_dist_type})",
            xaxis_title="Bid ($)", yaxis_title="Frequency",
            barmode="overlay", template="plotly_white", height=380, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with cx2:
        fig_low = go.Figure()
        fig_low.add_trace(go.Histogram(x=sim["lowest_comp"], nbinsx=50,
                                        marker_color="#264653", opacity=0.8,
                                        name="Lowest Competitor Bid"))
        fig_low.add_vline(x=miller_bid, line_dash="dash", line_color="blue", line_width=2,
                          annotation_text=f"Miller: ${miller_bid:,.0f}")
        fig_low.update_layout(
            title="Lowest Competitor Bid (Combined)",
            xaxis_title="Bid ($)", yaxis_title="Frequency",
            template="plotly_white", height=380, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_low, use_container_width=True)

# ── Tab 4: Data Summary ───────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Historical Data Summary")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**All Bid Preparation Costs**")
        st.write(f"- Records: {len(bid_prep_all)}")
        st.write(f"- Mean: ${bid_prep_all.mean():,.0f}")
        st.write(f"- Std Dev: ${bid_prep_all.std():,.0f}")
        st.write(f"- Min: ${bid_prep_all.min():,.0f}  |  Max: ${bid_prep_all.max():,.0f}")

    with d2:
        st.markdown("**Project Costs (won bids only)**")
        st.write(f"- Records: {len(project_costs_won)}")
        st.write(f"- Mean: ${project_costs_won.mean():,.0f}")
        st.write(f"- Std Dev: ${project_costs_won.std():,.0f}")
        st.write(f"- Min: ${project_costs_won.min():,.0f}  |  Max: ${project_costs_won.max():,.0f}")

    st.markdown("---")
    total_bids = len(bid_prep_all)
    won_bids = len(project_costs_won)
    st.write(f"**Historical win rate:** {won_bids} / {total_bids} = {won_bids/total_bids:.1%}")

    st.markdown("---")
    st.markdown("#### Raw Data")
    st.dataframe(uploaded_df, use_container_width=True, height=400)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:#8a9bb0; font-size:0.85rem; padding-bottom:1rem;">
    Miller Construction Co. – Contract Bidding Decision Support Tool &nbsp;|&nbsp;
    {n_sims:,} simulations &nbsp;|&nbsp;
    Bid Prep: {chosen_bp['Distribution']} &nbsp;|&nbsp;
    Project Cost: {chosen_pc['Distribution']} &nbsp;|&nbsp;
    Competitors: {comp_dist_type}
</div>
""", unsafe_allow_html=True)
