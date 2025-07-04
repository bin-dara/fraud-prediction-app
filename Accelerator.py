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
    menu = st.radio("Go to", ["EDA", "Visualization", "ML Prediction"])
    st.markdown("---")
    st.subheader("About Company")
    st.markdown("""
        We are a cutting-edge technology firm specializing in AI and data science solutions. 
        Our mission is to deliver secure, scalable, and innovative fraud detection tools 
        for the financial industry.
    """)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("financial_fraud_dataset.csv")
    return df

df = load_data()

# -----------------------------
# CLEAN DATA
# -----------------------------
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

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# EDA SECTION
# -----------------------------
if menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Class Distribution")
    class_counts = df[target_col].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    st.bar_chart(class_counts.set_index('Class'))

# -----------------------------
# VISUALIZATION SECTION
# -----------------------------
elif menu == "Visualization":
    st.title("📈 Visualization")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Not enough numeric columns for visualization.")
    else:
        x_axis = st.selectbox("X-Axis", num_cols)
        y_axis = st.selectbox("Y-Axis", num_cols)
        color_by = st.selectbox("Color By", [None] + list(df.columns))

        if st.button("Generate Scatter Plot"):
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig2 = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)

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

    