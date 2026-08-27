 

from pathlib import Path

import pandas as pd
import streamlit as st

 

st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide"
)

st.title("Data Overview")

st.write(
    """
    This page summarises the structure and quality of the
    loan-application dataset used throughout the analysis.
    It focuses on dataset dimensions, variable types, missing
    values, summary statistics and the loan-approval target.
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

    return df1, df2, data, duplicate_columns


 
# Load data
 

try:

    dataset_1, dataset_2, loan_data, duplicate_columns = (
        load_data()
    )

except Exception as error:

    st.error(
        "The loan datasets could not be loaded."
    )

    st.exception(error)

    st.stop()


 
 

if len(dataset_1) != len(dataset_2):

    st.warning(
        """
        The two source datasets contain different numbers of rows.
        The current merge assumes that both files describe the same
        applicants in the same row order.
        """
    )


 
# 1. Dataset Summary
 

st.subheader("1. Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Applicants",
        f"{loan_data.shape[0]:,}"
    )

with col2:

    st.metric(
        "Combined Variables",
        f"{loan_data.shape[1]}"
    )

with col3:

    st.metric(
        "Dataset 1 Variables",
        f"{dataset_1.shape[1]}"
    )

with col4:

    st.metric(
        "Dataset 2 Variables",
        f"{dataset_2.shape[1]}"
    )


st.caption(
    f"""
    The two source files contain information about the same
    applicants. {len(duplicate_columns)} overlapping column(s)
    were retained once during horizontal integration.
    """
)


 
# 2. Source Dataset Structure
 

st.subheader("2. Source Dataset Structure")

source_summary = pd.DataFrame({
    "Dataset": [
        "Loan Approval Data Set 1",
        "Loan Approval Data Set 2",
        "Integrated Dataset"
    ],

    "Rows": [
        dataset_1.shape[0],
        dataset_2.shape[0],
        loan_data.shape[0]
    ],

    "Columns": [
        dataset_1.shape[1],
        dataset_2.shape[1],
        loan_data.shape[1]
    ]
})


st.dataframe(
    source_summary,
    hide_index=True,
    use_container_width=True
)


if duplicate_columns:

    st.write(
        "**Overlapping columns identified during integration:**"
    )

    st.write(
        ", ".join(
            duplicate_columns
        )
    )


 
# 3. Loan Approval Distribution
 

st.subheader("3. Loan Approval Distribution")


if "LoanApproved" in loan_data.columns:

    approval_counts = (
        loan_data[
            "LoanApproved"
        ]
        .value_counts()
        .sort_index()
    )


    rejected_count = int(
        approval_counts.get(
            0,
            0
        )
    )

    approved_count = int(
        approval_counts.get(
            1,
            0
        )
    )

    total_target = (
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
                / total_target
                * 100
                if total_target > 0
                else 0
            ),

            (
                approved_count
                / total_target
                * 100
                if total_target > 0
                else 0
            )
        ]
    })


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
            .set_index(
                "Outcome"
            )[
                "Percentage"
            ]
        )


    if (
        approval_summary[
            "Percentage"
        ].max()
        >= 70
    ):

        st.info(
            """
            The target variable is imbalanced. Rejected applications
            represent a substantially larger proportion of the dataset
            than approved applications. This imbalance was considered
            during predictive model evaluation.
            """
        )


else:

    st.warning(
        "LoanApproved was not found in the integrated dataset."
    )


 
# 4. Missing-Value Summary
 
st.subheader("4. Missing-Value Summary")


missing_summary = pd.DataFrame({
    "Variable":
        loan_data.columns,

    "Missing_Count":
        loan_data
        .isna()
        .sum()
        .values,

    "Missing_Percent":
        (
            loan_data
            .isna()
            .mean()
            .values
            * 100
        )
})


missing_summary = (
    missing_summary
    .sort_values(
        [
            "Missing_Count",
            "Variable"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)


missing_only = (
    missing_summary[
        missing_summary[
            "Missing_Count"
        ] > 0
    ]
)


if len(
    missing_only
) > 0:

    st.dataframe(
        missing_only.round(2),
        hide_index=True,
        use_container_width=True
    )

    total_missing = int(
        loan_data
        .isna()
        .sum()
        .sum()
    )

    st.metric(
        "Total Missing Values",
        f"{total_missing:,}"
    )


else:

    st.success(
        "No missing values were detected in the integrated dataset."
    )

 
# 5. Variable Structure
 

st.subheader("5. Variable Structure")


variable_structure = pd.DataFrame({
    "Variable":
        loan_data.columns,

    "Data_Type":
        loan_data
        .dtypes
        .astype(str)
        .values,

    "Unique_Values": [
        loan_data[
            column
        ]
        .nunique(
            dropna=False
        )
        for column
        in loan_data.columns
    ],

    "Missing_Values":
        loan_data
        .isna()
        .sum()
        .values
})


st.dataframe(
    variable_structure,
    hide_index=True,
    use_container_width=True
)


 
# 6. Numerical Summary
 

st.subheader("6. Numerical Summary")


numerical_columns = (
    loan_data
    .select_dtypes(
        include="number"
    )
    .columns
)


if len(
    numerical_columns
) > 0:

    numerical_summary = (
        loan_data[
            numerical_columns
        ]
        .describe()
        .T
    )


    numerical_summary[
        "median"
    ] = (
        loan_data[
            numerical_columns
        ]
        .median()
    )


    numerical_summary = (
        numerical_summary[
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max"
            ]
        ]
    )


    st.dataframe(
        numerical_summary.round(2),
        use_container_width=True
    )


else:

    st.info(
        "No numerical variables were detected."
    )


 
# 7. Categorical Summary
 

st.subheader("7. Categorical Summary")


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


if categorical_columns:

    categorical_records = []


    for column in categorical_columns:

        value_counts = (
            loan_data[
                column
            ]
            .fillna(
                "Missing"
            )
            .astype(
                str
            )
            .value_counts()
        )


        dominant_category = (
            value_counts.index[0]
        )

        dominant_count = int(
            value_counts.iloc[0]
        )

        dominant_percent = (
            dominant_count
            / len(
                loan_data
            )
            * 100
        )


        categorical_records.append({
            "Variable":
                column,

            "Number_of_Categories":
                loan_data[
                    column
                ]
                .nunique(
                    dropna=False
                ),

            "Dominant_Category":
                dominant_category,

            "Dominant_Count":
                dominant_count,

            "Dominant_Percent":
                dominant_percent
        })


    categorical_summary = pd.DataFrame(
        categorical_records
    )


    st.dataframe(
        categorical_summary.round(2),
        hide_index=True,
        use_container_width=True
    )


else:

    st.info(
        "No analytical categorical variables were detected."
    )


 
# 8. Duplicate and Identifier Check
 

st.subheader("8. Structural Integrity")


if "ID" in loan_data.columns:

    duplicate_ids = int(
        loan_data[
            "ID"
        ]
        .duplicated()
        .sum()
    )

    unique_ids = int(
        loan_data[
            "ID"
        ]
        .nunique()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Unique Applicant IDs",
            f"{unique_ids:,}"
        )


    with col2:

        st.metric(
            "Duplicate IDs",
            f"{duplicate_ids:,}"
        )


    with col3:

        st.metric(
            "Total Applicants",
            f"{len(loan_data):,}"
        )


    if (
        duplicate_ids == 0
        and
        unique_ids == len(
            loan_data
        )
    ):

        st.success(
            "Applicant identifier integrity is preserved."
        )


else:

    st.info(
        "ID was not found in the integrated dataset."
    )


 
# 9. Dataset Preview
 

st.subheader("9. Dataset Preview")


preview_rows = st.slider(
    "Number of rows to display",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)


st.dataframe(
    loan_data.head(
        preview_rows
    ),
    use_container_width=True
)


 