# 🐠 Red Sea Fisheries Sustainability & Forecasting Tool

An independent data intelligence and predictive analytics platform built to harmonize multi-dimensional aquatic food system datasets, evaluate regional production thresholds, and simulate predictive fishing policy scenarios for the Red Sea basin.

---

##  Live Interactive Application
**[Click Here to Interact with the Live Streamlit Dashboard](https://red-sea-fisheries-intelligence-mpdfysmlem7npnrjfeoh7m.streamlit.app/)**
  
*(Explore interactive country multi-select filters, dynamic species sliders, and what-if machine learning policy simulators).*

---

##  Core Research Objectives & Methodology

This project builds an end-to-end data pipeline to address critical data engineering and analytical challenges in marine ecosystem management:

1. **Data Harmonization:** Integrating disparate raw data formats (unstructured capture quantities, taxonomic classifications, regional geographic boundaries, and socioeconomic income indices) into a single, unified analytical matrix.
2. **Identifying Knowledge Gaps:** Auditing data quality to expose regional reporting deficiencies, including missing reporting metadata flags and taxonomic resolution masking.
3. **Predictive Policy Modeling:** Developing an interpretative machine learning baseline to forecast fish stocks and interactively simulate conservation vs. exploitation pathways.

---

##  Key Analytical Insights

### 1. Data Completeness & Taxonomic Resolution Gaps
* **Metadata Gaps:** Over 90% of regional capture rows lack active statistical reporting metadata flags (`STATUS` anomalies).
* **Taxonomic Masking:** A major "Taxonomic Resolution Gap" was uncovered, where massive volumes of catch data are logged under broad categories (e.g., *Actinopterygii* or *Perciformes*) rather than specific species, obscuring exact biodiversity metrics.

### 2. Production Plateaus vs. Aquaculture Surges
* **Marine Stagnation:** Aggregated wild marine catches peaked in the early 2000s and have since plateaued, signaling that wild coral reef and marine fisheries may have hit their maximum ecological carrying capacity.
* **Aquaculture Reliance:** Recent 5-year trajectories reveal that overall regional landing growth is heavily driven by aquaculture channels (specifically *Oreochromis niloticus* in localized zones) rather than wild marine expansion.

### 3. Machine Learning Baseline Forecast (\(R^2 = 0.9022\))
* A baseline **Scikit-Learn Linear Regression** model predicts a steady historical growth velocity of **~10,406 tonnes/year**.
* Under static socio-economic variables, the model projects regional volumes to reach approximately **781,406 tonnes by 2026**.
* **Policy Simulation:** The application includes a custom slider allowing researchers to warp the regression slope to simulate aggressive catch quotas (conservation recovery) vs. unchecked exploitation.

---

##  Technical Stack & Tools

* **Core Language:** Python 3
* **Data Processing & Architecture:** Pandas, NumPy
* **Machine Learning & Estimation:** Scikit-Learn (LinearRegression, Metrics)
* **Data Visualization:** Matplotlib
* **User Interface & Cloud Deployment:** Streamlit Community Cloud

---

##  Repository File Architecture

```text
├── app.py                      # Main Streamlit application dashboard script
├── Red_Sea_Fisheries_Complete.ipynb       # Documented Jupyter Notebook showing full EDA & pipeline steps
├── capture_quantity.csv        # Primary landing volumes and temporal data
├── cl_fi_country_groups.csv    # Country codes mapped to regions & socioeconomic brackets
├── cl_fi_species_groups.csv     # Taxonomic codes mapped to scientific & common names
├── cl_fi_waterarea_groups.csv   # Area codes mapped to marine/inland classifications
├── fsj_unit.csv                 # Measurement unit multipliers and labels
└── README.md                   # Project documentation and summary
```

---

##  Local Installation & Execution

To run this data application on your local machine, clone this repository and follow these configuration steps:

1. **Clone the repository:**
   ```bash
   git clone github.com
   cd YOUR_REPO_NAME
   ```

2. **Install all required dependencies:**
   ```bash
   pip install streamlit pandas numpy matplotlib scikit-learn
   ```

3. **Launch the Streamlit web server:**
   ```bash
   streamlit run app.py
   ```
4. Open your browser and navigate to `http://localhost:8501`.
