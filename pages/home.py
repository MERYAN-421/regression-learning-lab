import streamlit as st

st.title("📈 Regression Learning Lab")

st.subheader(
    "Interactive Multiple Linear Regression "
    "& Statistical Diagnostics"
)

st.markdown(
    """
    Welcome to **Regression Learning Lab**.

    This platform helps students learn:

    - Multiple Linear Regression
    - Feature Selection
    - MSE, RMSE, and R²
    - Five Regression Assumptions
    - Model Comparison and Interpretation
    """
)

st.divider()

st.header("🎓 Learning Objectives")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**1. Data & Features**\n\n"
        "Explore datasets and select useful features."
    )

with col2:
    st.info(
        "**2. Regression Models**\n\n"
        "Train OLS models and evaluate predictions."
    )

with col3:
    st.info(
        "**3. Statistical Diagnostics**\n\n"
        "Visualize and investigate regression assumptions."
    )

st.divider()

st.header("🧭 Learning Workflow")

steps = [
    "1. Select a Dataset",
    "2. Explore and Preprocess Data",
    "3. Perform Feature Selection",
    "4. Train Multiple Linear Regression",
    "5. Evaluate Model Performance",
    "6. Diagnose Five Regression Assumptions",
    "7. Compare Experimental Results",
    "8. Generate a Learning Report"
]

for step in steps:
    st.write(step)

st.divider()

st.success(
    "🚀 Project Foundation initialized successfully!"
)

st.caption(
    "Version 0.1.0 | Milestone 0 — Project Foundation"
)