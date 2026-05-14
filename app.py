import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Page Configuration
st.set_page_config(page_title="Red Sea Fisheries Sustainability Tool", layout="wide", page_icon="🐠")

st.title("🐠 Red Sea Fisheries Sustainability & Forecasting Tool")
st.markdown("### 🤖 End-to-End Data Harmonization, Exploratory Analytics, and Predictive Policy Modeling")
st.markdown("---")




# Load Data (Cached to make the app fast)
@st.cache_data
def load_and_clean_data():
    #capture_df = pd.read_csv('capture_quantity.csv')
    #country_df = pd.read_csv('cl_fi_country_groups.csv')
    #species_df = pd.read_csv('cl_fi_species_groups.csv')
    # Fully resilient data parsers configured to bypass encoding blocks and skip corrupt markers
    capture_df = pd.read_csv('capture_quantity.csv', encoding='utf-8', encoding_errors='ignore')
    country_df = pd.read_csv('cl_fi_country_groups.csv', encoding='utf-8', encoding_errors='ignore')
    species_df = pd.read_csv('cl_fi_species_groups.csv', encoding='utf-8', encoding_errors='ignore')

    country_clean = country_df[
        ['UN_Code', 'ISO3_Code', 'Name_En', 'Continent_Group_En', 'EcoClass_Group_En', 'GeoRegion_Group_En']]
    master_df = pd.merge(capture_df, country_clean, left_on='COUNTRY.UN_CODE', right_on='UN_Code', how='inner')

    basin_countries = ['Saudi Arabia', 'Egypt', 'Sudan', 'Eritrea', 'Djibouti', 'Yemen', 'Jordan', 'Israel']
    red_sea_df = master_df[master_df['Name_En'].isin(basin_countries)].copy()

    species_clean = species_df[['3A_Code', 'Scientific_Name', 'Major_Group', 'ISSCAAP_Group_En']]
    red_sea_final = pd.merge(red_sea_df, species_clean, left_on='SPECIES.ALPHA_3_CODE', right_on='3A_Code', how='inner')
    return red_sea_final


try:
    raw_df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}. Ensure CSV files are in the same folder.")
    st.stop()

# --- SIDEBAR INTERACTIVE FILTERS ---
st.sidebar.header("📁 Navigation & Parameters")
app_mode = st.sidebar.selectbox("Choose the analysis phase:",
                                ["1. Project Overview & Knowledge Gaps", "2. Historical Trends & Species Analysis",
                                 "3. Interactive ML Policy Forecasting"])

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Geographic Filters")

all_countries = sorted(raw_df['Name_En'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select Red Sea Basin Countries to Analyze:",
    options=all_countries,
    default=all_countries
)

if not selected_countries:
    st.warning("Please select at least one country in the sidebar to populate data views.")
    st.stop()

df = raw_df[raw_df['Name_En'].isin(selected_countries)]

# --- PHASE 1: OVERVIEW & KNOWLEDGE GAPS ---
if app_mode == "1. Project Overview & Knowledge Gaps":
    st.header("🔍 Project Overview & Taxonomic Knowledge Gaps")

    col1, col2, col3 = st.columns(3)
    col1.metric("Subset Records Selected", f"{len(df):,}")
    col2.metric("Active Countries Filtered", f"{df['Name_En'].nunique()}")
    col3.metric("Missing Status Flags", f"{df['STATUS'].isnull().sum():,}")

    st.markdown("""
    ### 🔬 Identified Research Gaps
    * **Taxonomic Resolution Gap:** A significant volume of regional landing records are filed under broad groups like *Actinopterygii* instead of precise species names, masking true biodiversity metrics.
    * **Data Completeness Issues:** The high density of null cells in reporting flags implies gaps in baseline monitoring standards across specific coastal jurisdictions.
    """)
    st.dataframe(df.head(10))

# --- PHASE 2: HISTORICAL TRENDS & SPECIES ANALYSIS ---
elif app_mode == "2. Historical Trends & Species Analysis":
    st.header("📈 Historical Trends & Species Dynamics")

    # Global range filter for all charts in this tab
    yearly_trend = df.groupby('PERIOD')['VALUE'].sum()
    min_year, max_year = int(yearly_trend.index.min()), int(yearly_trend.index.max())
    selected_years = st.slider("Select Historical Range for Analysis:", min_year, max_year, (min_year, max_year))

    df_filtered = df[(df['PERIOD'] >= selected_years[0]) & (df['PERIOD'] <= selected_years[1])]

    # Split charts into logical tabs for structured reading
    tab1, tab2 = st.tabs(["🌍 Regional & Country Production Trends", "🐟 Species Diversity & Trajectories"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Chart 1: Aggregated Regional Production Over Time")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            trend_data = df_filtered.groupby('PERIOD')['VALUE'].sum()
            ax1.plot(trend_data.index, trend_data.values, color="teal", marker="o", linewidth=2)
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Production (Tonnes)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            st.caption(
                "Historical reality: Rapid expansion followed by a long-term production stabilization plateau post-2000.")

        with c2:
            st.subheader("Chart 2: Production Share Stacked Area Chart")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            country_yearly = df_filtered.groupby(['PERIOD', 'Name_En'])['VALUE'].sum().unstack().fillna(0)
            country_yearly.plot(kind='area', alpha=0.7, ax=ax2)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Production (Tonnes)')
            ax2.legend(title='Country', loc='upper left')
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig2)
            st.caption(
                "Geographic distribution showing Egypt's strong baseline dominance alongside stable outputs across other nations.")

    with tab2:
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Chart 3: Top Exploited Taxonomic Groups")
            top_n = st.slider("Select Number of Top Species to Rank:", 5, 15, 10)
            top_species = df_filtered.groupby('Scientific_Name')['VALUE'].sum().sort_values(ascending=False).head(top_n)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            top_species.plot(kind='barh', color='salmon', ax=ax3)
            ax3.set_xlabel('Total Production (Tonnes)')
            ax3.set_ylabel('Scientific Name')
            ax3.invert_yaxis()
            ax3.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig3)
            st.caption(
                "Taxonomic granularity checkpoint showing high reliance on broad groups alongside specialized catch data.")

        with c4:
            st.subheader("Chart 4: Recent Trajectories of Top 5 Dominant Taxa")
            # Isolate the top 5 historical groups globally
            global_top5 = df.groupby('Scientific_Name')['VALUE'].sum().sort_values(ascending=False).head(
                5).index.tolist()
            recent_trend_data = df_filtered[df_filtered['Scientific_Name'].isin(global_top5)]
            recent_pivot = recent_trend_data.groupby(['PERIOD', 'Scientific_Name'])['VALUE'].sum().unstack().fillna(0)

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            recent_pivot.plot(kind='line', marker='o', ax=ax4)
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Production (Tonnes)')
            ax4.legend(loc='upper left', fontsize='small')
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            st.caption(
                "Divergence trend tracking: Note aquaculture-driven variations contrasting with flatlining marine-only catch channels.")

# --- PHASE 3: INTERACTIVE ML POLICY FORECASTING ---
elif app_mode == "3. Interactive ML Policy Forecasting":
    st.header("🤖 Machine Learning Policy Forecasting Simulation")

    yearly_trend = df.groupby('PERIOD')['VALUE'].sum()
    X = yearly_trend.index.values.reshape(-1, 1)
    y = yearly_trend.values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    baseline_slope = model.coef_[0]

    st.markdown("### 🚦 Simulate Fishing Policy Scenarios")
    policy_multiplier = st.slider(
        "Policy Scenario Driver (1.0 = Historical Baseline Trend Velocity):",
        min_value=-2.0, max_value=2.0, value=1.0, step=0.1
    )

    forecast_horizon = st.slider("Years to Forecast into Future:", 1, 15, 5)
    future_years = np.array(range(int(X.max()) + 1, int(X.max()) + 1 + forecast_horizon)).reshape(-1, 1)

    simulated_slope = baseline_slope * policy_multiplier
    last_year = X.max()
    last_value = y[-1]

    simulated_predictions = []
    for yr in future_years.flatten():
        years_ahead = yr - last_year
        simulated_predictions.append(last_value + (simulated_slope * years_ahead))

    c1, c2, c3 = st.columns(3)
    c1.metric("Model $R^2$ Score (Goodness of Fit)", f"{r2:.4f}")
    c2.metric("Mean Absolute Error (MAE)", f"{mae:,.0f} t")

    if policy_multiplier < 0:
        c3.warning("🚨 Conservation / Stock Recovery Scenario Simulated")
    elif policy_multiplier == 1.0:
        c3.info("⚖️ Standard Historical Trend Line")
    else:
        c3.error("💥 High Exploitation Scenario Simulated")

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(X, y, label='Historical Landing Record', color='blue', marker='o')
    ax5.plot(future_years, simulated_predictions,
             label=f'Policy Simulation Forecast (Slope Multiplier: x{policy_multiplier})', color='red', linestyle='--',
             marker='s')
    ax5.set_title("Simulated Fishing Trajectory Forward Projection")
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Catch (Tonnes)")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    st.pyplot(fig5)
