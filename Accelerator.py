import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import base64

# -----------------------------
# PAGE CONFIG & BACKGROUND COLOR
# -----------------------------
st.set_page_config(page_title="Financial Fraud Detection", page_icon="💰", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #EBEEF8;
    }}
    .sidebar .sidebar-content {{
        background-color: #f0f2f6;
    }}
    h1, h2, h3 {{
        color: #0e4c92;
    }}
    .stButton>button {{
        background-color: #0e4c92;
        color: white;
    }}
    .css-1aumxhk {{
        padding: 2rem;
    }}
        
 

    
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# COMPANY LOGO & ABOUT SECTION
# -----------------------------
COMPANY_LOGO_URL = "https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png"

with st.sidebar:
    st.image(COMPANY_LOGO_URL, use_column_width=True)
    st.title("Navigation")
    menu = st.radio("", ["Visualization", "ML Prediction"])
    st.markdown("---")
    st.subheader("About Company")
    st.markdown("""
        We are a cutting-edge technology firm specializing in AI and data science solutions. 
        Our mission is to deliver secure, scalable, and innovative fraud detection tools 
        for the financial industry.
        
    """)
    st.markdown("---")
    st.subheader("Connect With Us")
    st.markdown(
        """
        <div style="display: flex; gap: 10px; align-items: center;">
            <a href="https://booleandata.ai/" target="_blank">🌐</a>
            <a href="https://www.facebook.com/Booleandata" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24">
            </a>
            <a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24">
            </a>
            <a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# LOAD DATA
# -----------------------------
# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"financial_fraud_dataset.csv")
    return df

df = load_data()


# -----------------------------
# CLEAN & PREPROCESS DATA
# -----------------------------
@st.cache_data
def preprocess_data(df):
    # Drop unwanted columns
    drop_cols = [col for col in df.columns if 'unnamed' in col.lower() or 'id' in col.lower() or 'timestamp' in col.lower()]
    df = df.drop(columns=drop_cols)

    # Guess target as last column
    target_col = df.columns[-1]

    # Label encode target if categorical
    if df[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])
    else:
        le_target = None

    # Separate features and target
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical columns
    cat_cols = X_raw.select_dtypes(include='object').columns.tolist()

    # Label encode categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        label_encoders[col] = le

    return df, X_raw, y, target_col, le_target, cat_cols, label_encoders

df, X_raw, y, target_col, le_target, cat_cols, label_encoders = preprocess_data(df)


# -----------------------------
# TRAIN MODEL (cached)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy

clf, accuracy = train_model(X_raw, y)



# -----------------------------
# VISUALIZATION SECTION
# -----------------------------
if menu == "Visualization":
    st.title("📈 Visualization")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    
    st.subheader("Pie Chart - Amount Vs Fraud")
    pie_data = df.groupby("Is_fraud")["Amount"].sum().reset_index()
    pie_colors = [
    # "#08306b",  # very dark blue
    # "#08519c",  # dark
    # "#2171b5",  # medium-dark
        "#4292c6",  # medium
        "#6baed6",  # medium-light
        "#9ecae1",  # light
        "#c6dbef"   # very light
    ]

    pie_fig = px.pie(
        pie_data,
        names="Is_fraud",
        values="Amount",
        color_discrete_sequence=pie_colors
    )
    pie_fig.update_traces(hole=0, textinfo="percent+label")
    st.plotly_chart(pie_fig, use_container_width=True)


    st.subheader("Correlation Heatmap")
    fig2 = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)


    if "Transaction_type" in df.columns and "Amount" in df.columns:
            # Group by Transaction Type
            summary = df.groupby("Transaction_type")["Amount"].sum().reset_index()



    st.subheader("Donut Chart - Amount Vs Fraud")        # ---------------------------
            # Donut Chart
            # ---------------------------
    donut_colors = [
        #"#08306b",  # very dark blue
        "#08519c",  # dark
        "#2171b5",  # medium-dark
        "#4292c6",  # medium
        "#6baed6",  # medium-light
        "#9ecae1",  # light
        "#c6dbef"   # very light
    ]



    fig = px.pie(
        summary,
        names="Transaction_type",
        values="Amount",
        hole=0.5,
        color_discrete_sequence=donut_colors
    )

    st.plotly_chart(fig, use_container_width=True)

    # Check columns
    if "Is_high _risk_country" in df.columns and "Location" in df.columns:
            # Clean/Standardize
            df["Is_high_risk_country"] = df["Is_high_risk_country"].astype(str)
            df["Location"] = df["Location"].astype(str)

            # Group data
            area_data = df.groupby(["Location", "Is_high_risk_country"]).size().reset_index(name="Count")

            # Area chart using Plotly
            fig = px.area(
                area_data,
                x="Location",
                y="Risk",
                color="Is High Risk Country",
                title="High Risk Country by Location (Area Chart)",
                color_discrete_sequence=px.colors.sequential.dense  # dark blue-style theme
            )

            fig.update_layout(
                plot_bgcolor="#F9F9F9",
                paper_bgcolor="#F9F9F9",
                title_font_color="darkblue",
                font_color="black"
            )

            st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ML PREDICTION SECTION
# -----------------------------
elif menu == "ML Prediction":
    st.title("🤖 ML Prediction")
    st.markdown("Fill in the form below to predict **Fraud** or **Not Fraud**")

    with st.form("prediction_form"):
        st.subheader("Input Features")
        user_input = {}
        for col in X_raw.columns:
            if col in cat_cols:
                options = sorted(df[col].unique())
                user_input[col] = st.selectbox(f"{col}", options)
            else:
                val = int(df[col].mean())
                user_input[col] = st.number_input(f"{col}", value=val)

        submitted = st.form_submit_button("Predict")

    if submitted:
     input_df = pd.DataFrame([user_input])

    # Encode categoricals
    for col in cat_cols:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except:
            st.error(f"Invalid value for {col}. Please choose a valid option.")
            st.stop()

    prediction = clf.predict(input_df)[0]
    pred_proba = clf.predict_proba(input_df)[0]

    if le_target:
        prediction_label = le_target.inverse_transform([prediction])[0]
    else:
        prediction_label = str(prediction)

    fraud_percent = pred_proba[1] * 100
    not_fraud_percent = pred_proba[0] * 100

    st.success(f"""
    ### 🎯 Model Prediction
    - Predicted Class: **{prediction_label}**
    - ✅ Fraud Probability: **{fraud_percent:.2f}%**
    - ✅ Not Fraud Probability: **{not_fraud_percent:.2f}%**
    """)

    st.subheader("🔎 Model Test Accuracy")
    st.write(f"✅ {accuracy*100:.2f}%")

    
