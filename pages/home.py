import streamlit as st

# Header & Branding
st.title("📈 Regression Learning Lab")
st.subheader("Interactive Multiple Linear Regression & Statistical Diagnostics Platform")

st.markdown(
    """
    **Regression Learning Lab** is an interactive educational platform designed for undergraduate and graduate
    students in data science, statistics, and machine learning. Through curated real-world Kaggle regression
    datasets (14–39 features), learners gain hands-on intuition for the end-to-end regression modeling
    lifecycle—from exploratory analysis and feature selection to rigorous statistical hypothesis testing and diagnostic analysis.
    """
)

st.divider()

# Core Learning Objectives
st.header("🎓 Learning Objectives")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### 1. Data & Features
        - Understand feature distributions and multicollinearity.
        - Handle missing values and scale continuous variables.
        - Compare filtering, wrapper, and embedded feature selection techniques.
        """
    )

with col2:
    st.info(
        """
        ### 2. Regression Modeling
        - Master Ordinary Least Squares (OLS) estimation.
        - Interpret coefficients, p-values, and confidence intervals.
        - Evaluate predictive power using MSE, RMSE, R², and Adjusted R².
        """
    )

with col3:
    st.info(
        """
        ### 3. Statistical Diagnostics
        - Verify the **Five Classical Gauss-Markov Assumptions**.
        - Detect leverage points, outliers, and influential observations.
        - Understand model limitations and remediation strategies.
        """
    )

st.divider()

# Learning Modes
st.header("🧭 Learning Modes")

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    with st.container(border=True):
        st.markdown("### 📘 Guided Learning Mode")
        st.caption("Recommended for beginners and structured coursework")
        st.markdown(
            """
            - **Structured Progression**: Step-by-step guidance through the standard 8-stage data science workflow.
            - **Educational Checkpoints**: Explanations for statistical assumptions, test hypotheses ($H_0$ / $H_1$), and rule-of-thumb thresholds.
            - **Interpreted Diagnostic Feedback**: Plain-language summaries of potential assumption violations.
            """
        )

with col_mode2:
    with st.container(border=True):
        st.markdown("### 🔬 Free Experiment Mode")
        st.caption("Recommended for advanced analysis and open-ended lab experiments")
        st.markdown(
            """
            - **Custom Workflows**: Freely toggle feature subsets, transformation parameters, and train/test splits.
            - **Multi-Model Sandbox**: Run and save multiple regression trials side-by-side.
            - **Exportable Artifacts**: Compare performance across experiments and generate reproducible analytical reports.
            """
        )

st.divider()

# End-to-End Workflow
st.header("🔄 End-to-End Learning Workflow")

workflow_cols = st.columns(4)

with workflow_cols[0]:
    with st.container(border=True):
        st.markdown("#### Phase 1: Ingestion")
        st.markdown("**1. Dataset Library**\nSelect curated Kaggle benchmark data.")
        st.markdown("**2. Data Exploration**\nInspect distributions, correlations, and anomalies.")

with workflow_cols[1]:
    with st.container(border=True):
        st.markdown("#### Phase 2: Selection")
        st.markdown("**3. Feature Selection**\nApply Pearson, F-test, MI, RFE, SFS, or Lasso.")
        st.markdown("**4. Regression Model**\nFit OLS and inspect coefficient statistics.")

with workflow_cols[2]:
    with st.container(border=True):
        st.markdown("#### Phase 3: Validation")
        st.markdown("**5. Model Evaluation**\nMeasure MSE, RMSE, R², and residual patterns.")
        st.markdown("**6. Diagnostics**\nTest the 5 classical regression assumptions.")

with workflow_cols[3]:
    with st.container(border=True):
        st.markdown("#### Phase 4: Synthesis")
        st.markdown("**7. Experiment Comparison**\nCompare multiple runs and feature sets.")
        st.markdown("**8. Learning Report**\nGenerate summary findings and export report.")

st.divider()

# Platform Status & Roadmap
st.header("🚀 System Status & Milestone Roadmap")

st.success(
    """
    **Current Active Milestone: Milestone 0 — Project Foundation**
    
    The modular project foundation, Streamlit navigation system, UI theme, documentation, and page routing
    are fully initialized. Machine learning algorithms, Kaggle datasets, and statistical diagnostic tests will be implemented in subsequent milestones.
    """
)

roadmap_data = [
    {"Milestone": "Milestone 0", "Scope": "Project Foundation, Multi-page Navigation, UI/UX Architecture", "Status": "✅ Completed"},
    {"Milestone": "Milestone 1", "Scope": "Dataset Library, Preprocessing, Config Schema (14–39 features)", "Status": "⏳ Scheduled"},
    {"Milestone": "Milestone 2", "Scope": "Exploratory Data Analysis (EDA) & Feature Selection Suite", "Status": "⏳ Scheduled"},
    {"Milestone": "Milestone 3", "Scope": "Multiple Linear Regression (OLS) & Model Evaluation", "Status": "⏳ Scheduled"},
    {"Milestone": "Milestone 4", "Scope": "Five Regression Assumptions & Statistical Diagnostic Tests", "Status": "⏳ Scheduled"},
    {"Milestone": "Milestone 5", "Scope": "Experiment Comparison, History Tracking & Learning Report", "Status": "⏳ Scheduled"},
]

st.table(roadmap_data)

st.caption("Regression Learning Lab | Version 0.1.0 | Academic Edition")