import streamlit as st
import base64
from churn_model import train_model
from utils import plot_roc, plot_distribution, plot_class_pie

def set_background(image_file: str):
    with open(image_file, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main > div {{
            background: rgba(255,255,255,0.9);
            padding: 2rem;
            border-radius: 10px;
            margin-top: 3rem;
        }}
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Churn Predictor", layout="centered")
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    set_background("background_home.jpg")
    st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    if st.button("🚀 Get Started"):
        st.session_state.page = "predict"
        st.rerun()

elif st.session_state.page == "predict":
    set_background("background_upload.jpg")
    st.title("📤 Upload Your Telco Churn CSV")
    uploaded = st.file_uploader("", type=["csv"])

    if uploaded:
        with open("uploaded_file.csv", "wb") as f:
            f.write(uploaded.getbuffer())

        st.markdown(
            """
            <div style='background-color:#ffffff; padding:10px 20px; border-radius:10px; font-weight:bold; color:#000000; display:inline-block;'>
                🔄 Training model...
            </div>
            """, unsafe_allow_html=True
        )

        model, preds_pd, auc_val = train_model("uploaded_file.csv")
        st.session_state.page = "results"
        st.session_state.model = model
        st.session_state.preds_pd = preds_pd
        st.session_state.auc_val = auc_val
        st.rerun()

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "results":
    set_background("background_upload.jpg")
    st.header("📊 Churn Prediction Results")

    st.markdown(
        f"""
        <div style='background-color:#ffffff; padding:10px 20px; border-radius:10px; font-weight:bold; color:#000000; display:inline-block;'>
            ✅ Trained! Test AUC = {st.session_state.auc_val:.4f}
        </div>
        """, unsafe_allow_html=True
    )

    st.subheader("📈 ROC Curve")
    st.pyplot(plot_roc(st.session_state.preds_pd))

    st.subheader("📊 Churn Probability Distribution")
    st.pyplot(plot_distribution(st.session_state.preds_pd))

    st.subheader("📉 Class Distribution Pie")
    st.pyplot(plot_class_pie(st.session_state.preds_pd))

    st.subheader("📋 Sample Predictions")
    st.dataframe(st.session_state.preds_pd[["label", "prediction", "churn_prob"]].head(20))

    if st.button("🔙 Back to Upload & Predict"):
        st.session_state.page = "predict"
        st.rerun()
