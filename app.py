import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessor
#model = joblib.load("models/rf_model.pkl")
#preprocessor = joblib.load("models/preprocessor.pkl")

model = joblib.load("models/rf_model.pkl")  # This now includes preprocessor


# Page setup
st.set_page_config(page_title="Industrial Machine Anomaly Detection", layout="centered")

# App title in header
st.markdown("<h1 style='text-align: center;'>Employee Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Go to", ["Home", "EDA"])

# ---------- PAGE: HOME ----------
if page == "Home":
    st.subheader("Enter Employee Details for Salary Prediction")

    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 75, 30)
            education = st.selectbox("Education Level ", [
                    'School',
                    'High School Graduate',
                    'Associate Degree',
                    'Bachelor’s Degree',
                    'Master’s Degree',
                    'Doctorate'
                ])
            hours_per_week = st.slider("Working Hours/Week", 1, 100, 40)
            experience = st.slider("Years of Experience", 0, 40)

        with col2:
            occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                                      'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                                      'Armed-Forces'])
            relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                                         'Other-relative', 'Unmarried'])
            native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany',
                                                             'Canada', 'India', 'England', 'China', 'Other'])

        submit = st.form_submit_button("Predict")

        education_mapping = {
                            'School': 1,
                            'High School Graduate': 9,
                            'Associate Degree': 12,
                            'Bachelor’s Degree': 14,
                            'Master’s Degree': 15,
                            'Doctorate': 16
                        }

    if submit:
        input_dict = {
            'age': age,
            'education': education_mapping[education],
            'hours_per_week': hours_per_week,
            'occupation': occupation,
            'relationship': relationship,
            'native_country': native_country,
            'experience': experience
        }

        input_df = pd.DataFrame([input_dict])
       # processed_input = preprocessor.transform(input_df)
       # prediction = model.predict(processed_input)
        
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df).max()
        
        proba = model.predict_proba(input_df).max()

        salary_class = ">50K" if prediction[0] == 1 else "≤50K"

        st.success(f"💰 Predicted Salary: **{salary_class}**")
        st.info(f"📊 Confidence: **{round(proba * 100, 2)}%**")
        st.caption("🧾 Note: Education levels range from School (1) to Doctorate (16).")

# ---------- PAGE: EDA ----------
elif page == "EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    try:
        data = pd.read_csv("data/raw/adult.csv")

        if st.checkbox("Show Raw Data"):
            st.write(data.head())

        st.write("### Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data["age"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### Hours per Week vs Salary Class")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=data, x="income", y="hours.per.week", ax=ax2)
        st.pyplot(fig2)

        st.write("### Correlation Heatmap")
        corr = data.select_dtypes(include=np.number).corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

    except FileNotFoundError:
        st.error("📁 `cleaned_salary_data.csv` not found in `data/` folder.")

# ---------- FOOTER ----------
st.markdown("""<hr style='margin-top: 40px;'>
    <p style='text-align: center; font-size: 14px;'>
    Rajeshwari Sonawane | Microsoft AI & Azure Internship 2025
    </p>""", unsafe_allow_html=True)

