 

from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

 

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("Loan Approval Modelling and Prediction")

st.write(
    """
    Enter the applicant and loan information below and select
    one of the trained classification models developed in Task 3.
    """
)


 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

 
# Load trained objects
 

@st.cache_resource
def load_model_artifacts():

    artifacts = {
        "preprocessor": joblib.load(
            MODEL_DIR / "preprocessor.joblib"
        ),

        "baseline_logistic": joblib.load(
            MODEL_DIR / "baseline_logistic.joblib"
        ),

        "final_logistic": joblib.load(
            MODEL_DIR / "final_logistic.joblib"
        ),

        "random_forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),

        "optimal_threshold": joblib.load(
            MODEL_DIR / "optimal_threshold.joblib"
        ),

        "feature_metadata": joblib.load(
            MODEL_DIR / "feature_metadata.joblib"
        ),

        "deployment_config": joblib.load(
            MODEL_DIR / "deployment_config.joblib"
        )
    }

    return artifacts


try:

    artifacts = load_model_artifacts()

except Exception as error:

    st.error(
        "The trained model artifacts could not be loaded."
    )

    st.exception(error)

    st.stop()


preprocessor = artifacts["preprocessor"]

baseline_logistic = artifacts[
    "baseline_logistic"
]

final_logistic = artifacts[
    "final_logistic"
]

random_forest = artifacts[
    "random_forest"
]

optimal_threshold = float(
    artifacts["optimal_threshold"]
)

deployment_config = artifacts[
    "deployment_config"
]

winsor_bounds = deployment_config[
    "winsor_bounds"
]

expected_input_features = deployment_config[
    "preprocessor_input_features"
]


 
# Model selection
 

st.subheader("1. Select Prediction Model")

model_choice = st.selectbox(
    "Prediction model",
    [
        "Threshold-Optimised Logistic Regression",
        "Baseline Logistic Regression",
        "Random Forest"
    ]
)


if model_choice == (
    "Threshold-Optimised Logistic Regression"
):

    st.success(
        """
        **Recommended model**

        This model uses the training-derived decision threshold
        of 0.47 instead of the default 0.50.

        Final test performance:

        - Accuracy: 99.52%
        - Precision: 99.06%
        - Recall: 98.95%
        - F1-score: 99.01%
        """
    )

elif model_choice == "Baseline Logistic Regression":

    st.info(
        """
        Baseline Logistic Regression uses the conventional
        classification threshold of 0.50.

        Test accuracy: 99.50%
        """
    )

else:

    st.info(
        """
        Random Forest provides a nonlinear alternative to
        Logistic Regression.

        Test accuracy: 98.72%
        """
    )


 
# Applicant input form
 

st.subheader("2. Applicant and Loan Information")

with st.form("loan_prediction_form"):

  

    st.markdown("### Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        application_date = st.date_input(
            "Application Date",
            value=date.today()
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
            step=1
        )

        experience = st.number_input(
            "Experience (years)",
            min_value=0,
            max_value=80,
            value=15,
            step=1
        )

    with col2:

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Employed",
                "Self-Employed",
                "Unemployed"
            ]
        )

        education_level = st.selectbox(
            "Education Level",
            [
                "High School",
                "Associate",
                "Bachelor",
                "Master",
                "Doctorate"
            ]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced",
                "Widowed",
                "Unknown"
            ]
        )

    with col3:

        home_ownership = st.selectbox(
            "Home Ownership Status",
            [
                "Mortgage",
                "Rent",
                "Own",
                "Other"
            ]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Home",
                "Debt Consolidation",
                "Auto",
                "Education",
                "Other"
            ]
        )

        number_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )


 

    st.markdown("### Income and Financial Position")

    col1, col2, col3 = st.columns(3)

    with col1:

        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

        checking_balance = st.number_input(
            "Checking Account Balance",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )

        savings_balance = st.number_input(
            "Savings Account Balance",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

    with col2:

        total_assets = st.number_input(
            "Total Assets",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )

        total_liabilities = st.number_input(
            "Total Liabilities",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )

        net_worth = st.number_input(
            "Net Worth",
            min_value=0.0,
            value=70000.0,
            step=1000.0
        )

    with col3:

        monthly_debt_payments = st.number_input(
            "Existing Monthly Debt Payments",
            min_value=0.0,
            value=400.0,
            step=25.0
        )

        job_tenure = st.number_input(
            "Job Tenure (years)",
            min_value=0,
            max_value=60,
            value=5,
            step=1
        )


   
    # Loan information
     

    st.markdown("### Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1.0,
            value=25000.0,
            step=500.0
        )

        loan_duration = st.number_input(
            "Loan Duration (months)",
            min_value=1,
            max_value=180,
            value=48,
            step=12
        )

    with col2:

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.01,
            max_value=100.0,
            value=23.0,
            step=0.1
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=250,
            max_value=900,
            value=580,
            step=1
        )

    with col3:

        credit_utilisation = st.number_input(
            "Credit Card Utilisation Rate",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        credit_history_length = st.number_input(
            "Length of Credit History (years)",
            min_value=0,
            max_value=80,
            value=15,
            step=1
        )


 
    # Credit behaviour
   

    st.markdown("### Credit Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:

        open_credit_lines = st.number_input(
            "Number of Open Credit Lines",
            min_value=0,
            max_value=50,
            value=3,
            step=1
        )

        credit_inquiries = st.number_input(
            "Number of Credit Inquiries",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

    with col2:

        payment_history = st.number_input(
            "Payment History",
            min_value=0.0,
            value=24.0,
            step=1.0
        )

        bankruptcy_history = st.selectbox(
            "Bankruptcy History",
            [
                "No",
                "Yes"
            ]
        )

    with col3:

        previous_defaults = st.selectbox(
            "Previous Loan Defaults",
            [
                "No",
                "Yes"
            ]
        )


  
    # Risk Score
 

    st.markdown("### Risk Assessment")

    risk_score_missing = st.checkbox(
        "RiskScore is unavailable / missing"
    )

    risk_score = st.number_input(
        "Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        disabled=risk_score_missing
    )


   

    submitted = st.form_submit_button(
        "Generate Prediction",
        type="primary"
    )


 

if submitted:

    try:

      

        applicant = {
            "AnnualIncome":
                float(annual_income),

            "EmploymentStatus":
                employment_status,

            "Experience":
                float(experience),

            "LoanDuration":
                float(loan_duration),

            "NumberOfDependents":
                int(number_dependents),

            "MonthlyDebtPayments":
                float(monthly_debt_payments),

            "NumberOfOpenCreditLines":
                int(open_credit_lines),

            "BankruptcyHistory":
                1 if bankruptcy_history == "Yes" else 0,

            "PreviousLoanDefaults":
                1 if previous_defaults == "Yes" else 0,

            "LengthOfCreditHistory":
                float(credit_history_length),

            "CheckingAccountBalance":
                float(checking_balance),

            "TotalLiabilities":
                float(total_liabilities),

            "NetWorth":
                float(net_worth),

            "Age":
                float(age),

            "CreditScore":
                float(credit_score),

            "LoanAmount":
                float(loan_amount),

            "CreditCardUtilizationRate":
                float(credit_utilisation),

            "NumberOfCreditInquiries":
                int(credit_inquiries),

            "PaymentHistory":
                float(payment_history),

            "SavingsAccountBalance":
                float(savings_balance),

            "TotalAssets":
                float(total_assets),

            "JobTenure":
                float(job_tenure),

            "InterestRate":
                float(interest_rate),

            "EducationLevel":
                education_level,

            "MaritalStatus":
                marital_status,

            "HomeOwnershipStatus":
                home_ownership,

            "LoanPurpose":
                loan_purpose,

            "RiskScore":
                np.nan
                if risk_score_missing
                else float(risk_score),

            "RiskScoreMissing":
                1 if risk_score_missing else 0
        }


        
   
        

        applicant["ApplicationYear"] = (
            application_date.year
        )

        applicant["ApplicationMonth"] = (
            application_date.month
        )

        applicant["ApplicationDay"] = (
            application_date.day
        )

        applicant["ApplicationDayOfWeek"] = (
            application_date.weekday()
        )


   

        monthly_income = (
            applicant["AnnualIncome"] / 12
        )

        monthly_rate = (
            applicant["InterestRate"]
            / 100
            / 12
        )

        number_of_payments = (
            applicant["LoanDuration"]
        )

        principal = (
            applicant["LoanAmount"]
        )


        if monthly_rate > 0:

            monthly_loan_payment = (
                principal
                * (
                    monthly_rate
                    * (1 + monthly_rate)
                    ** number_of_payments
                )
                /
                (
                    (1 + monthly_rate)
                    ** number_of_payments
                    - 1
                )
            )

        else:

            monthly_loan_payment = (
                principal
                / number_of_payments
            )


        total_debt_to_income = (
            (
                monthly_loan_payment
                + applicant[
                    "MonthlyDebtPayments"
                ]
            )
            /
            monthly_income
            * 100
        )


        applicant["MonthlyIncome"] = (
            monthly_income
        )

        applicant["MonthlyLoanPayment"] = (
            monthly_loan_payment
        )

        applicant["TotalDebtToIncomeRatio"] = (
            total_debt_to_income
        )


       

        model_input = pd.DataFrame(
            [applicant]
        )


        

        for feature, bounds in (
            winsor_bounds.items()
        ):

            if feature in model_input.columns:

                model_input[feature] = (
                    model_input[feature]
                    .clip(
                        lower=bounds["lower"],
                        upper=bounds["upper"]
                    )
                )


   

        missing_features = [
            feature
            for feature in expected_input_features
            if feature not in model_input.columns
        ]

        if missing_features:

            raise ValueError(
                "Application input is missing "
                f"required features: {missing_features}"
            )


        model_input = model_input[
            expected_input_features
        ]


     

        transformed_input = (
            preprocessor.transform(
                model_input
            )
        )


        
        # Generate prediction
       

        if model_choice == (
            "Baseline Logistic Regression"
        ):

            model = baseline_logistic

            probability = (
                model.predict_proba(
                    transformed_input
                )[0, 1]
            )

            threshold = 0.50


        elif model_choice == (
            "Threshold-Optimised "
            "Logistic Regression"
        ):

            model = final_logistic

            probability = (
                model.predict_proba(
                    transformed_input
                )[0, 1]
            )

            threshold = (
                optimal_threshold
            )


        else:

            model = random_forest

            probability = (
                model.predict_proba(
                    transformed_input
                )[0, 1]
            )

            threshold = 0.50


        prediction = int(
            probability >= threshold
        )


    
        # Display results
        

        st.divider()

        st.subheader("3. Prediction Result")

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )


        with result_col1:

            if prediction == 1:

                st.success(
                    "LOAN APPROVED"
                )

            else:

                st.error(
                    "LOAN REJECTED"
                )


        with result_col2:

            st.metric(
                "Approval Probability",
                f"{probability * 100:.2f}%"
            )


        with result_col3:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )


  

        st.progress(
            float(
                np.clip(
                    probability,
                    0,
                    1
                )
            )
        )


        

        with st.expander(
            "View engineered financial features"
        ):

            engineered_summary = pd.DataFrame({
                "Feature": [
                    "MonthlyIncome",
                    "MonthlyLoanPayment",
                    "TotalDebtToIncomeRatio"
                ],

                "Value": [
                    monthly_income,
                    monthly_loan_payment,
                    total_debt_to_income
                ]
            })

            st.dataframe(
                engineered_summary,
                use_container_width=True,
                hide_index=True
            )


      
        # Model explanation
       

        st.caption(
            f"Prediction generated using: "
            f"{model_choice}"
        )

        st.warning(
            """
            This prediction represents the output of the
            analytical model developed for this project.
            It should not be interpreted as a real-world
            lending decision.
            """
        )


    except Exception as error:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(error)