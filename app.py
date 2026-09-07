import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

BACKGROUND_IMAGE = os.path.join(SRC_DIR, "bg.png")

CLASSIFICATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "regression_model.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "classification_label_encoder.pkl"
)


# ============================================================
# BACKGROUND IMAGE (cached so it isn't re-read/encoded on every rerun)
# ============================================================

@st.cache_data
def get_base64_image(image_path):

    if not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode()


background_base64 = get_base64_image(BACKGROUND_IMAGE)


# ============================================================
# CUSTOM DESIGN
# ============================================================

def apply_design():

    if background_base64:

        background_css = f"""
        <style>

        /* ================================
           MAIN APPLICATION BACKGROUND
        ================================= */

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(7, 20, 42, 0.48),
                    rgba(7, 20, 42, 0.48)
                ),
                url("data:image/png;base64,{background_base64}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """

        st.markdown(
            background_css,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <style>

        /* ==================================================
           GENERAL
        ================================================== */

        .main {
            background: transparent !important;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        /* Remove Streamlit default bordered containers */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }


        /* ==================================================
           SIDEBAR
        ================================================== */

        section[data-testid="stSidebar"] {
            background: rgba(3, 15, 34, 0.96) !important;
            border-right: 1px solid rgba(255,255,255,0.12);
        }

        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
        }


        /* ==================================================
           HEADINGS
        ================================================== */

        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }

        h2 {
            color: #ffffff !important;
            font-weight: 750 !important;
        }

        h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        p {
            color: #f4f7fb;
        }


        /* ==================================================
           LABELS
        ================================================== */

        label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        div[data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }


        /* ==================================================
           TEXT INPUT
        ================================================== */

        div[data-testid="stTextInput"] input {
            background-color: rgba(255,255,255,0.96) !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            border: 1px solid rgba(255,255,255,0.8) !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #6b7280 !important;
            -webkit-text-fill-color: #6b7280 !important;
        }


        /* ==================================================
           NUMBER INPUT
        ================================================== */

        div[data-testid="stNumberInput"] {
            color: #111827 !important;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInputContainer"] input,
        div[data-testid="stNumberInput"] input[type="number"] {
            background-color: rgba(255,255,255,0.96) !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
            border: none !important;
            font-weight: 600 !important;
        }

        div[data-testid="stNumberInput"] input::placeholder {
            color: #6b7280 !important;
            -webkit-text-fill-color: #6b7280 !important;
        }

        div[data-testid="stNumberInput"] button {
            background-color: #e5e7eb !important;
            color: #111827 !important;
            border: none !important;
        }

        /* Catch-all fallback: some Streamlit versions render the number
           input's inner element without the above test-ids picking it up,
           which otherwise leaves white text on a white field. */
        input[type="number"],
        input[type="text"] {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }


        /* ==================================================
           SELECT BOX (closed state)
        ================================================== */

        div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.97) !important;
            border-radius: 10px !important;
            min-height: 44px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.97) !important;
            border: none !important;
            border-radius: 10px !important;
        }

        /* Force every descendant (span, div, p, whatever the version
           renders the selected value as) to dark text — the closed
           box was showing white-on-white because only `span` was
           targeted before, and some versions render the value in a
           plain div instead. */
        div[data-baseweb="select"] * {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        div[data-baseweb="select"] svg {
            fill: #111827 !important;
        }


        /* ==================================================
           SELECT BOX (open dropdown / popover)
        ================================================== */
        /* The dropdown list renders in a portal (data-baseweb="popover"),
           so it needs its own background — otherwise it shows the
           page's dark background behind white-ish text. */

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        ul[role="listbox"] {
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] li,
        ul[role="listbox"] li,
        li[role="option"] {
            color: #111827 !important;
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] li *,
        ul[role="listbox"] li *,
        li[role="option"] * {
            color: #111827 !important;
        }

        div[data-baseweb="popover"] li:hover,
        ul[role="listbox"] li:hover,
        li[role="option"]:hover,
        li[aria-selected="true"] {
            background-color: #eef4ff !important;
            color: #111827 !important;
        }


        /* ==================================================
           SLIDER
        ================================================== */

        div[data-testid="stSlider"] {
            color: #ffffff !important;
        }


        /* ==================================================
           FORM CONTAINER (keep transparent like the rest of the page)
        ================================================== */

        div[data-testid="stForm"] {
            background: transparent !important;
            border: none !important;
        }


        /* ==================================================
           BUTTONS
        ================================================== */

        div.stButton > button,
        div.stFormSubmitButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 10px;
            border: none;
            background: linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );
            color: white !important;
            font-size: 16px;
            font-weight: 700;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        }


        /* ==================================================
           METRIC CARDS
        ================================================== */

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px;
            padding: 18px;
            backdrop-filter: blur(8px);
        }

        div[data-testid="stMetricLabel"] {
            color: #dbeafe !important;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 800 !important;
        }


        /* ==================================================
           SUCCESS / WARNING / ERROR
        ================================================== */

        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            font-weight: 600;
        }


        /* ==================================================
           EXPANDER
        ================================================== */

        details {
            background: rgba(255,255,255,0.10) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 12px !important;
        }

        details summary {
            color: #ffffff !important;
            font-weight: 700 !important;
        }


        /* ==================================================
           DIVIDER
        ================================================== */

        hr {
            border-color: rgba(255,255,255,0.22) !important;
        }


        /* ==================================================
           RADIO BUTTONS
        ================================================== */

        div[role="radiogroup"] label {
            color: #ffffff !important;
        }


        /* ==================================================
           CAPTION
        ================================================== */

        .stCaption {
            color: #e5e7eb !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


apply_design()


# ============================================================
# LOAD MODELS (cached so they load once, not on every rerun)
# ============================================================

@st.cache_resource
def load_models():

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    return classification_model, regression_model, label_encoder


try:

    classification_model, regression_model, label_encoder = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Model loading failed. Please check the files inside the models folder."
    )

    st.code(str(e))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 💳 EMIPredict AI")

    st.caption(
        "Intelligent Financial Risk Assessment"
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "💳 EMI Eligibility",
            "💰 Maximum EMI",
            "📊 Model Information"
        ]
    )

    st.markdown("---")

    st.markdown("### 🤖 Machine Learning")

    st.caption("🔬 MLflow Experiment Tracking")

    st.caption("🌐 Streamlit Application")


# ============================================================
# INPUT FUNCTION
# ============================================================
#
# Wrapped in st.form so that filling in fields does NOT rerun the
# whole app on every keystroke/selection — the app only reruns once,
# when the submit button is pressed.
# ============================================================

def get_customer_inputs(form_key, submit_label):

    with st.form(key=form_key):

        # --------------------------------------------------------
        # PERSONAL INFORMATION
        # --------------------------------------------------------

        st.markdown("## 👤 Personal Information")

        st.caption(
            "Enter the customer's basic personal details."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col3:

            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married"]
            )

        col1, col2 = st.columns(2)

        with col1:

            education = st.selectbox(
                "Education",
                [
                    "High School",
                    "Graduate",
                    "Post Graduate",
                    "Professional"
                ]
            )

        with col2:

            family_size = st.number_input(
                "Family Size",
                min_value=1,
                max_value=20,
                value=4,
                step=1
            )


        st.markdown("---")


        # --------------------------------------------------------
        # EMPLOYMENT AND INCOME
        # --------------------------------------------------------

        st.markdown("## 💼 Employment & Income")

        st.caption(
            "Enter employment and monthly income information."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            monthly_salary = st.number_input(
                "Monthly Salary (₹)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f"
            )

        with col2:

            employment_type = st.selectbox(
                "Employment Type",
                [
                    "Private",
                    "Government",
                    "Self-employed"
                ]
            )

        with col3:

            years_of_employment = st.number_input(
                "Years of Employment",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5
            )

        col1, col2 = st.columns(2)

        with col1:

            company_type = st.selectbox(
                "Company Type",
                [
                    "Small",
                    "Medium",
                    "Large"
                ]
            )

        with col2:

            house_type = st.selectbox(
                "House Type",
                [
                    "Rented",
                    "Own",
                    "Family"
                ]
            )


        st.markdown("---")


        # --------------------------------------------------------
        # HOUSEHOLD EXPENSES
        # --------------------------------------------------------

        st.markdown("## 🏠 Household Expenses")

        st.caption(
            "Enter monthly household expenses."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            monthly_rent = st.number_input(
                "Monthly Rent (₹)",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                format="%.2f"
            )

        with col2:

            dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=20,
                value=2,
                step=1
            )

        with col3:

            school_fees = st.number_input(
                "School Fees (₹)",
                min_value=0.0,
                value=3000.0,
                step=500.0,
                format="%.2f"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            college_fees = st.number_input(
                "College Fees (₹)",
                min_value=0.0,
                value=2000.0,
                step=500.0,
                format="%.2f"
            )

        with col2:

            travel_expenses = st.number_input(
                "Travel Expenses (₹)",
                min_value=0.0,
                value=3000.0,
                step=500.0,
                format="%.2f"
            )

        with col3:

            groceries_utilities = st.number_input(
                "Groceries & Utilities (₹)",
                min_value=0.0,
                value=8000.0,
                step=500.0,
                format="%.2f"
            )

        col1, col2 = st.columns(2)

        with col1:

            other_monthly_expenses = st.number_input(
                "Other Monthly Expenses (₹)",
                min_value=0.0,
                value=3000.0,
                step=500.0,
                format="%.2f"
            )

        with col2:

            existing_loans = st.selectbox(
                "Existing Loans",
                ["Yes", "No"]
            )


        st.markdown("---")


        # --------------------------------------------------------
        # FINANCIAL INFORMATION
        # --------------------------------------------------------

        st.markdown("## 💰 Financial Information")

        st.caption(
            "Enter current debt and financial stability details."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            current_emi_amount = st.number_input(
                "Current EMI Amount (₹)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                format="%.2f"
            )

        with col2:

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=700,
                step=1
            )

        with col3:

            bank_balance = st.number_input(
                "Bank Balance (₹)",
                min_value=0.0,
                value=100000.0,
                step=5000.0,
                format="%.2f"
            )

        col1, col2 = st.columns(2)

        with col1:

            emergency_fund = st.number_input(
                "Emergency Fund (₹)",
                min_value=0.0,
                value=50000.0,
                step=5000.0,
                format="%.2f"
            )

        with col2:

            emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI"
                ]
            )


        st.markdown("---")


        # --------------------------------------------------------
        # REQUEST DETAILS
        # --------------------------------------------------------

        st.markdown("## 📋 EMI Request Details")

        col1, col2 = st.columns(2)

        with col1:

            requested_amount = st.number_input(
                "Requested Loan / EMI Amount (₹)",
                min_value=0.0,
                value=100000.0,
                step=5000.0,
                format="%.2f"
            )

        with col2:

            requested_tenure = st.number_input(
                "Requested Tenure (Months)",
                min_value=1,
                max_value=360,
                value=24,
                step=1
            )


        st.markdown("---")

        submitted = st.form_submit_button(
            submit_label,
            use_container_width=True
        )


    if not submitted:
        return None, False


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "marital_status": [marital_status],

        "education": [education],

        "monthly_salary": [monthly_salary],

        "employment_type": [employment_type],

        "years_of_employment": [years_of_employment],

        "company_type": [company_type],

        "house_type": [house_type],

        "monthly_rent": [monthly_rent],

        "family_size": [family_size],

        "dependents": [dependents],

        "school_fees": [school_fees],

        "college_fees": [college_fees],

        "travel_expenses": [travel_expenses],

        "groceries_utilities": [groceries_utilities],

        "other_monthly_expenses": [
            other_monthly_expenses
        ],

        "existing_loans": [existing_loans],

        "current_emi_amount": [
            current_emi_amount
        ],

        "credit_score": [credit_score],

        "bank_balance": [bank_balance],

        "emergency_fund": [emergency_fund],

        "emi_scenario": [emi_scenario],

        "requested_amount": [
            requested_amount
        ],

        "requested_tenure": [
            requested_tenure
        ]
    })


    # ========================================================
    # ENGINEERED FEATURES
    # ========================================================

    salary = input_data["monthly_salary"].replace(
        0,
        np.nan
    )


    # EMI-to-Income Ratio

    input_data["emi_to_income_ratio"] = (
        input_data["current_emi_amount"] /
        salary
    )


    # Rent-to-Income Ratio

    input_data["rent_to_income_ratio"] = (
        input_data["monthly_rent"] /
        salary
    )


    # Total monthly expenses

    total_expenses = (

        input_data["school_fees"]

        + input_data["college_fees"]

        + input_data["travel_expenses"]

        + input_data["groceries_utilities"]

        + input_data["other_monthly_expenses"]
    )


    # Expense-to-Income Ratio

    input_data["expense_to_income_ratio"] = (
        total_expenses /
        salary
    )


    # Debt-to-Income Ratio

    input_data["debt_to_income_ratio"] = (
        input_data["current_emi_amount"] /
        salary
    )


    # Handle infinity and missing values

    input_data = input_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    input_data = input_data.fillna(0)


    return input_data, True


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("💳 EMIPredict AI")

    st.subheader(
        "Smart EMI Prediction & Financial Risk Assessment"
    )

    st.write(
        "Use machine learning to evaluate EMI eligibility "
        "and estimate the maximum monthly EMI a customer "
        "can safely afford."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🤖 ML Models",
            "6"
        )

    with col2:

        st.metric(
            "📊 Prediction Types",
            "2"
        )

    with col3:

        st.metric(
            "⚡ Real-Time",
            "Yes"
        )

    st.markdown("---")

    st.subheader("🚀 What Can You Do?")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### 💳 EMI Eligibility")

        st.write(
            "Predict whether a customer is eligible "
            "for the requested EMI based on their "
            "financial profile."
        )

    with col2:

        st.write("### 💰 Maximum EMI")

        st.write(
            "Estimate the maximum monthly EMI that "
            "the customer can safely afford."
        )

    st.markdown("---")

    st.info(
        "💡 Navigate using the sidebar to start a prediction."
    )


# ============================================================
# EMI ELIGIBILITY PAGE
# ============================================================

elif page == "💳 EMI Eligibility":

    st.title("💳 EMI Eligibility Prediction")

    st.write(
        "Evaluate whether the customer is eligible "
        "for the requested EMI."
    )

    st.markdown("---")

    input_data, submitted = get_customer_inputs(
        form_key="eligibility_form",
        submit_label="🔍 Predict EMI Eligibility"
    )

    if submitted:

        if not models_loaded:

            st.error(
                "Models are not loaded. Please check the models folder."
            )

        else:

            try:

                prediction = classification_model.predict(
                    input_data
                )

                # Handle encoded classification output

                try:

                    prediction_label = label_encoder.inverse_transform(
                        prediction.astype(int)
                    )[0]

                except Exception:

                    prediction_label = str(
                        prediction[0]
                    )


                st.markdown("---")

                st.subheader(
                    "📊 Prediction Result"
                )

                result_text = str(
                    prediction_label
                ).lower()


                if (
                    "eligible" in result_text
                    and "not" not in result_text
                ):

                    st.success(
                        f"✅ Customer is **{prediction_label}**"
                    )

                elif (
                    "not" in result_text
                    or "reject" in result_text
                    or "ineligible" in result_text
                ):

                    st.error(
                        f"❌ Customer is **{prediction_label}**"
                    )

                else:

                    st.info(
                        f"📌 Prediction: **{prediction_label}**"
                    )


            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(e)
                )


# ============================================================
# MAXIMUM EMI PAGE
# ============================================================

elif page == "💰 Maximum EMI":

    st.title("💰 Maximum EMI Prediction")

    st.write(
        "Estimate the maximum monthly EMI the customer "
        "can safely afford."
    )

    st.markdown("---")

    input_data, submitted = get_customer_inputs(
        form_key="max_emi_form",
        submit_label="💰 Predict Maximum EMI"
    )

    if submitted:

        if not models_loaded:

            st.error(
                "Models are not loaded. Please check the models folder."
            )

        else:

            try:

                prediction = regression_model.predict(
                    input_data
                )

                max_emi = float(
                    prediction[0]
                )

                st.markdown("---")

                st.subheader(
                    "📊 Maximum Affordable EMI"
                )

                st.metric(
                    "Maximum Monthly EMI",
                    f"₹{max_emi:,.2f}"
                )

                if max_emi > 0:

                    st.success(
                        "The predicted EMI represents "
                        "the estimated maximum monthly "
                        "EMI based on the customer's "
                        "financial profile."
                    )

                else:

                    st.warning(
                        "The model predicted a very low "
                        "or zero affordable EMI."
                    )


            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(e)
                )


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "📊 Model Information":

    st.title("📊 Model Information")

    st.write(
        "EMIPredict AI uses machine learning models "
        "for classification and regression."
    )

    st.markdown("---")


    st.subheader("🤖 Classification")

    st.write(
        "The classification model predicts EMI eligibility."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Type",
            "Classifier"
        )

    with col2:
        st.metric(
            "Prediction",
            "Eligibility"
        )

    with col3:
        st.metric(
            "Evaluation",
            "Accuracy / F1"
        )


    st.markdown("---")


    st.subheader("💰 Regression")

    st.write(
        "The regression model predicts the maximum "
        "monthly EMI the customer can afford."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Type",
            "Regressor"
        )

    with col2:
        st.metric(
            "Prediction",
            "Maximum EMI"
        )

    with col3:
        st.metric(
            "Evaluation",
            "RMSE / MAE / R²"
        )


    st.markdown("---")


    st.subheader("⚙️ Engineered Features")

    st.write(
        "The application calculates the following "
        "financial ratios before prediction:"
    )

    st.write(
        "• EMI-to-Income Ratio"
    )

    st.write(
        "• Rent-to-Income Ratio"
    )

    st.write(
        "• Expense-to-Income Ratio"
    )

    st.write(
        "• Debt-to-Income Ratio"
    )


    st.markdown("---")


    st.subheader("🔬 Machine Learning Workflow")

    st.write(
        "Data → Cleaning → EDA → Feature Engineering "
        "→ Model Training → Evaluation → MLflow "
        "→ Streamlit Prediction"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EMIPredict AI • Machine Learning Based EMI Risk Assessment"
)