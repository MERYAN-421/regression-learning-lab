import streamlit as st

# Global website configuration
st.set_page_config(
    page_title="Regression Learning Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define website navigation with official st.Page
pages = {
    "Introduction": [
        st.Page(
            "pages/home.py",
            title="Home",
            icon="🏠",
            default=True
        )
    ],
    "Regression Workflow": [
        st.Page(
            "pages/dataset.py",
            title="Dataset Library",
            icon="📂"
        ),
        st.Page(
            "pages/exploration.py",
            title="Data Exploration",
            icon="📊"
        ),
        st.Page(
            "pages/feature_selection.py",
            title="Feature Selection",
            icon="🔍"
        ),
        st.Page(
            "pages/regression.py",
            title="Regression Model",
            icon="📈"
        ),
        st.Page(
            "pages/evaluation.py",
            title="Model Evaluation",
            icon="🎯"
        ),
        st.Page(
            "pages/diagnostics.py",
            title="Assumption Diagnostics",
            icon="🧪"
        )
    ],
    "Results": [
        st.Page(
            "pages/comparison.py",
            title="Experiment Comparison",
            icon="⚖️"
        ),
        st.Page(
            "pages/report.py",
            title="Learning Report",
            icon="📝"
        )
    ]
}

# Render sidebar navigation
pg = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("📈 Regression Learning Lab")
    st.caption("Version 0.1.0 | Academic Edition")
    st.caption("Multiple Linear Regression Education")

# Run selected page
pg.run()
