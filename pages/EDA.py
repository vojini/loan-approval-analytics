 

from pathlib import Path

import pandas as pd
import streamlit as st


 

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("Exploratory Data Analysis")

st.write(
    """
    This page provides interactive exploratory analysis of the
    loan-application dataset. The focus is on distributions,
    relationships and approval patterns identified during Task 2.
    """
)


 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

DATASET_1 = DATA_DIR / "Loan_Approval_Data_Set_1.csv"
DATASET_2 = DATA_DIR / "Loan_Approval_Data_Set_2.csv"


 
# Load and combine datasets
 

@st.cache_data
def load_data():

    df1 = pd.read_csv(DATASET_1)
    df2 = pd.read_csv(DATASET_2)

    # Both files describe the same applicants.
    # Remove overlapping fields from Dataset 2 before
    # combining horizontally.
    duplicate_columns = [
        column
        for column in df2.columns
        if column in df1.columns
    ]

    df2_unique = df2.drop(
        columns=duplicate_columns
    )

    data = pd.concat(
        [
            df1.reset_index(drop=True),
            df2_unique.reset_index(drop=True)
        ],
        axis=1
    )

    return data


 
# Cached correlation calculation
 

@st.cache_data
def calculate_target_correlations(data):

    numeric_data = (
        data
        .select_dtypes(include="number")
    )

    if "LoanApproved" not in numeric_data.columns:
        return pd.DataFrame()

    correlations = (
        numeric_data
        .corr()["LoanApproved"]
        .drop("LoanApproved")
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False
        )
    )

    return pd.DataFrame({
        "Variable":
            correlations.index,

        "Correlation":
            correlations.values,

        "Absolute_Correlation":
            correlations.abs().values
    })


 
# Load data
 

try:

    loan_data = load_data()

except Exception as error:

    st.error(
        "The loan datasets could not be loaded."
    )

    st.exception(error)

    st.stop()


# 
# Define analytical variable groups
 

numerical_columns = (
    loan_data
    .select_dtypes(include="number")
    .columns
    .tolist()
)

numerical_features = [
    column
    for column in numerical_columns
    if column != "LoanApproved"
]


 
categorical_columns = [
    column
    for column in (
        loan_data
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )
    if column not in [
        "ID",
        "ApplicationDate"
    ]
]


 
# 1. Loan Approval Overview
 

st.subheader("1. Loan Approval Overview")

if "LoanApproved" in loan_data.columns:

    approval_counts = (
        loan_data["LoanApproved"]
        .value_counts()
        .sort_index()
    )

    rejected_count = int(
        approval_counts.get(0, 0)
    )

    approved_count = int(
        approval_counts.get(1, 0)
    )

    total_count = (
        rejected_count
        + approved_count
    )

    approval_summary = pd.DataFrame({
        "Outcome": [
            "Rejected",
            "Approved"
        ],

        "Count": [
            rejected_count,
            approved_count
        ],

        "Percentage": [
            (
                rejected_count
                / total_count
                * 100
                if total_count > 0
                else 0
            ),
            (
                approved_count
                / total_count
                * 100
                if total_count > 0
                else 0
            )
        ]
    })


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Applications",
            f"{total_count:,}"
        )

    with col2:

        st.metric(
            "Rejected",
            f"{rejected_count:,}"
        )

    with col3:

        st.metric(
            "Approved",
            f"{approved_count:,}"
        )


    left_col, right_col = st.columns(
        [1, 2]
    )

    with left_col:

        st.dataframe(
            approval_summary.round(2),
            hide_index=True,
            use_container_width=True
        )

    with right_col:

        st.bar_chart(
            approval_summary
            .set_index("Outcome")[
                "Percentage"
            ]
        )


    st.info(
        """
        The dataset contains substantially more rejected than
        approved applications. This class imbalance was considered
        during model evaluation using precision, recall and F1-score
        in addition to accuracy.
        """
    )


 
# 2. Numerical Distribution Explorer
 

st.subheader("2. Numerical Distribution Explorer")

default_numeric_index = 0

if "AnnualIncome" in numerical_features:

    default_numeric_index = (
        numerical_features.index(
            "AnnualIncome"
        )
    )


selected_numeric = st.selectbox(
    "Select a numerical variable",
    numerical_features,
    index=default_numeric_index
)


numeric_series = (
    loan_data[selected_numeric]
    .dropna()
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Mean",
        f"{numeric_series.mean():,.2f}"
    )

with col2:

    st.metric(
        "Median",
        f"{numeric_series.median():,.2f}"
    )

with col3:

    st.metric(
        "Std. Dev.",
        f"{numeric_series.std():,.2f}"
    )

with col4:

    st.metric(
        "Missing",
        int(
            loan_data[
                selected_numeric
            ]
            .isna()
            .sum()
        )
    )


# Histogram using 20 bins
histogram_counts = pd.cut(
    numeric_series,
    bins=20,
    include_lowest=True
).value_counts().sort_index()


histogram_df = pd.DataFrame({
    "Range": [
        str(interval)
        for interval
        in histogram_counts.index
    ],

    "Count":
        histogram_counts.values
}).set_index(
    "Range"
)


st.bar_chart(
    histogram_df
)


 
# 3. Categorical Distribution Explorer
 

st.subheader("3. Categorical Distribution Explorer")


if categorical_columns:

    default_category_index = 0

    if "EmploymentStatus" in categorical_columns:

        default_category_index = (
            categorical_columns.index(
                "EmploymentStatus"
            )
        )


    selected_category = st.selectbox(
        "Select a categorical variable",
        categorical_columns,
        index=default_category_index,
        key="categorical_distribution"
    )


    category_counts = (
        loan_data[
            selected_category
        ]
        .fillna("Missing")
        .astype(str)
        .value_counts()
    )


    category_summary = pd.DataFrame({
        "Category":
            category_counts.index,

        "Count":
            category_counts.values,

        "Percentage":
            (
                category_counts.values
                / len(loan_data)
                * 100
            )
    })


    left_col, right_col = st.columns(
        [1, 2]
    )

    with left_col:

        st.dataframe(
            category_summary.round(2),
            hide_index=True,
            use_container_width=True
        )

    with right_col:

        st.bar_chart(
            category_summary
            .set_index("Category")[
                "Percentage"
            ]
        )


else:

    st.info(
        "No suitable categorical variables were detected."
    )

 
# 4. Numerical Relationship Explorer
 

st.subheader("4. Numerical Relationship Explorer")


relationship_col1, relationship_col2 = (
    st.columns(2)
)


default_x_index = 0

if "CreditScore" in numerical_features:

    default_x_index = (
        numerical_features.index(
            "CreditScore"
        )
    )


default_y_index = min(
    1,
    len(numerical_features) - 1
)

if "InterestRate" in numerical_features:

    default_y_index = (
        numerical_features.index(
            "InterestRate"
        )
    )


with relationship_col1:

    x_variable = st.selectbox(
        "X-axis variable",
        numerical_features,
        index=default_x_index,
        key="relationship_x"
    )


with relationship_col2:

    y_variable = st.selectbox(
        "Y-axis variable",
        numerical_features,
        index=default_y_index,
        key="relationship_y"
    )


if x_variable != y_variable:

    relationship_data = (
        loan_data[
            [
                x_variable,
                y_variable
            ]
        ]
        .dropna()
    )


    correlation = (
        relationship_data
        .corr()
        .iloc[0, 1]
    )


    st.metric(
        "Pearson Correlation",
        f"{correlation:.3f}"
    )


 
    if len(
        relationship_data
    ) > 1500:

        relationship_plot_data = (
            relationship_data.sample(
                1500,
                random_state=42
            )
        )

    else:

        relationship_plot_data = (
            relationship_data
        )


    st.scatter_chart(
        relationship_plot_data,
        x=x_variable,
        y=y_variable
    )


    if abs(correlation) >= 0.70:

        st.write(
            "**Interpretation:** Strong linear relationship."
        )

    elif abs(correlation) >= 0.40:

        st.write(
            "**Interpretation:** Moderate linear relationship."
        )

    elif abs(correlation) >= 0.20:

        st.write(
            "**Interpretation:** Weak-to-moderate linear relationship."
        )

    else:

        st.write(
            "**Interpretation:** Weak linear relationship."
        )


else:

    st.warning(
        "Select two different numerical variables."
    )


 
# 5. Numerical Correlation with Loan Approval
 

st.subheader(
    "5. Numerical Correlation with Loan Approval"
)


correlation_table = (
    calculate_target_correlations(
        loan_data
    )
)


if not correlation_table.empty:

    maximum_top_n = min(
        15,
        len(correlation_table)
    )


    top_n = st.slider(
        "Number of variables to display",
        min_value=5,
        max_value=maximum_top_n,
        value=min(
            10,
            maximum_top_n
        )
    )


    top_correlations = (
        correlation_table
        .head(top_n)
    )


    left_col, right_col = st.columns(
        [1, 2]
    )

    with left_col:

        st.dataframe(
            top_correlations[
                [
                    "Variable",
                    "Correlation"
                ]
            ]
            .round(3),
            hide_index=True,
            use_container_width=True
        )

    with right_col:

        correlation_chart = (
            top_correlations[
                [
                    "Variable",
                    "Correlation"
                ]
            ]
            .set_index(
                "Variable"
            )
        )

        st.bar_chart(
            correlation_chart
        )


 
# 6. Approval Rate by Category
 

st.subheader(
    "6. Approval Rate by Category"
)


if (
    categorical_columns
    and
    "LoanApproved"
    in loan_data.columns
):

    default_approval_index = 0

    if "EducationLevel" in categorical_columns:

        default_approval_index = (
            categorical_columns.index(
                "EducationLevel"
            )
        )


    approval_category = st.selectbox(
        "Select categorical variable",
        categorical_columns,
        index=default_approval_index,
        key="approval_category"
    )


    category_for_approval = (
        loan_data[
            approval_category
        ]
        .fillna("Missing")
        .astype(str)
    )


    approval_analysis_data = pd.DataFrame({
        "Category":
            category_for_approval,

        "LoanApproved":
            loan_data[
                "LoanApproved"
            ]
    })


    category_approval = (
        approval_analysis_data
        .groupby(
            "Category"
        )[
            "LoanApproved"
        ]
        .agg(
            Applications="count",
            Approval_Rate="mean"
        )
        .reset_index()
    )


    category_approval[
        "Approval_Rate_Percent"
    ] = (
        category_approval[
            "Approval_Rate"
        ]
        * 100
    )


    category_approval = (
        category_approval
        .sort_values(
            "Approval_Rate_Percent",
            ascending=False
        )
    )


    left_col, right_col = st.columns(
        [1, 2]
    )

    with left_col:

        st.dataframe(
            category_approval[
                [
                    "Category",
                    "Applications",
                    "Approval_Rate_Percent"
                ]
            ]
            .round(2),
            hide_index=True,
            use_container_width=True
        )


    with right_col:

        approval_chart = (
            category_approval[
                [
                    "Category",
                    "Approval_Rate_Percent"
                ]
            ]
            .set_index(
                "Category"
            )
        )

        st.bar_chart(
            approval_chart
        )


    overall_approval_rate = (
        loan_data[
            "LoanApproved"
        ]
        .mean()
        * 100
    )


    st.caption(
        f"Overall approval rate: "
        f"{overall_approval_rate:.2f}%"
    )


 
# 7. Key EDA Findings
 

st.subheader(
    "7. Key EDA Findings"
)


st.markdown(
    """
    - **AnnualIncome** showed the strongest positive numerical
      relationship with loan approval.
    - **InterestRate** and **LoanAmount** were negatively associated
      with approval.
    - **EducationLevel** demonstrated the clearest categorical
      association with approval.
    - **Age** and **Experience** were highly correlated.
    - **NetWorth** and **TotalAssets** also showed substantial
      predictor redundancy.
    - Several financial variables exhibited strong positive skewness
      and extreme upper-tail observations.
    """
)


 