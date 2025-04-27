# app.py

# --------------------------
# Import Libraries
# --------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --------------------------
# Train a Basic Model
# --------------------------
# Dummy dataset
X = np.random.rand(200, 3)  # 3 features
y = np.random.randint(0, 2, 200)  # 0 = Low Risk, 1 = High Risk

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------
# Streamlit App Starts
# --------------------------
st.set_page_config(page_title="Heart Health Checkup", page_icon="🫀")
st.title('🫀 Heart Attack Risk Prediction Portal')

st.write('---')

# --------------------------
# Step 1: Basic Information
# --------------------------
st.subheader('📝 Enter Your Basic Information')

name = st.text_input('Your Name')
age = st.number_input('Your Age', min_value=1, max_value=120, step=1)
gender = st.selectbox('Select Gender', ['Male', 'Female', 'Other'])

if name and age:
    st.success(f"Hello {name}! Let's check your heart health. ❤️")

    st.write('---')

    # --------------------------
    # Step 2: Show Ideal Values
    # --------------------------
    st.subheader('📋 Ideal Health Parameters')

    ideal_heart_rate = 70  # average ideal value
    ideal_blood_sugar = 100  # mg/dL
    ideal_cholesterol = 180  # mg/dL

    st.info(f"Ideal Heart Rate: {ideal_heart_rate} bpm")
    st.info(f"Ideal Blood Sugar Level: {ideal_blood_sugar} mg/dL")
    st.info(f"Ideal Cholesterol Level: {ideal_cholesterol} mg/dL")

    st.write('---')

    # --------------------------
    # Step 3: Enter Actual Values
    # --------------------------
    st.subheader('🔎 Enter Your Health Checkup Results')

    user_heart_rate = st.slider('Your Heart Rate (bpm)', 30, 200, 70)
    user_blood_sugar = st.slider('Your Blood Sugar Level (mg/dL)', 50, 300, 100)
    user_cholesterol = st.slider('Your Cholesterol Level (mg/dL)', 100, 400, 180)

    st.write('---')

    # --------------------------
    # Step 4: Compare Actual vs Ideal
    # --------------------------
    st.subheader('📊 Your Health Report')

    compare_data = pd.DataFrame({
        'Parameters': ['Heart Rate', 'Blood Sugar', 'Cholesterol'],
        'Ideal': [ideal_heart_rate, ideal_blood_sugar, ideal_cholesterol],
        'Your Value': [user_heart_rate, user_blood_sugar, user_cholesterol]
    })

    st.dataframe(compare_data.set_index('Parameters'))

    # Bar Chart
    fig, ax = plt.subplots()
    index = np.arange(len(compare_data))
    bar_width = 0.35

    ax.bar(index, compare_data['Ideal'], bar_width, label='Ideal', color='green')
    ax.bar(index + bar_width, compare_data['Your Value'], bar_width, label='Your Value', color='red')

    ax.set_xlabel('Parameters')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Ideal vs Your Health Values')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(compare_data['Parameters'])
    ax.legend()

    st.pyplot(fig)

    st.write('---')

    # --------------------------
    # Step 5: Predict Risk
    # --------------------------
    st.subheader('🚑 Heart Attack Risk Prediction')

    if st.button('🔍 Predict My Heart Health'):
        input_features = np.array([[user_heart_rate, user_blood_sugar, user_cholesterol]])
        prediction = model.predict(input_features)

        if prediction[0] == 1:
            st.error('⚠️ High Risk of Heart Attack!')
            st.markdown('😟 Please consult your doctor immediately.')
        else:
            st.success('✅ Low Risk of Heart Attack!')
            st.markdown('😄 Keep maintaining your healthy lifestyle!')

    st.write('---')

    st.caption('Made with ❤️ by [Your Name]')

else:
    st.warning('👆 Please enter your Name and Age to continue.')

