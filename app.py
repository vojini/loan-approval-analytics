 

import streamlit as st

 
# Page configuration
 

st.set_page_config(
    page_title="Loan Approval Analysis",
    page_icon="💰",
    layout="wide"
)


 
# Main title
 

st.title("Loan Approval Analysis Application")

st.markdown(
    """
    This multipage application provides an interactive interface
    for exploring the loan-application dataset and using the
    machine-learning models developed during the analysis.
    """
)


 
# Project overview
 

st.subheader("Application Overview")

st.write(
    """
    The application contains three main sections:

    - **Modelling and Prediction** — enter applicant information,
      select a trained model, and generate a loan-approval prediction.

    - **Data Overview** — review dataset structure, dimensions,
      variables, missing values, and loan-approval distribution.

    - **Exploratory Data Analysis (EDA)** — explore selected
      statistical summaries and visualisations from the analysis.
    """
)


 

st.subheader("Final Classification Model")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Selected Model**")
    st.markdown(
        """
        <div style="
            font-size: 22px;
            font-weight: 600;
            padding-top: 1px;
        ">
            Logistic Regression
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.metric(
        label="Test Accuracy",
        value="99.52%"
    )

with col3:
    st.metric(
        label="F1-Score",
        value="99.01%"
    )

with col4:
    st.metric(
        label="Decision Threshold",
        value="0.47"
    )

 
# Model improvement explanation
 

st.subheader("Model Improvement")

st.info(
    """
    The final Logistic Regression model uses a decision threshold
    of 0.47 instead of the default 0.50.

    The threshold was selected using training-only cross-validation
    and improved test accuracy from 99.50% to 99.52%, while also
    improving recall and F1-score.
    """
)


 

st.subheader("Available Prediction Models")

st.markdown(
    """
    The modelling page provides access to:

    1. **Baseline Logistic Regression**
    2. **Threshold-Optimised Logistic Regression**
    3. **Random Forest**

    The threshold-optimised Logistic Regression model is the
    recommended model based on the final evaluation.
    """
)

 

 


 

 