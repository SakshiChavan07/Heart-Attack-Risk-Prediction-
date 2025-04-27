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
# Train the Model
# --------------------------
# Create dummy dataset
X = np.random.rand(200, 3)  # 3 features: age, heart rate, blood sugar
y = np.random.randint(0, 2, 200)  # 0 = Low Risk, 1 = High Risk

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------
# Streamlit Web App
# --------------------------
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️")
st.title('❤️ Heart Attack Risk Prediction App')

st.write("""
## Enter your health information:
""")

# Take input from user
age = st.slider('Select your Age:', 1, 120, 30)
heart_rate = st.slider('Select your Heart Rate:', 30, 220, 70)
blood_sugar = st.slider('Select your Blood Sugar Level:', 50, 300, 100)

# Show Bar Chart of Inputs
st.subheader('📊 Your Health Parameters:')
health_data = pd.DataFrame({
    'Parameter': ['Age', 'Heart Rate', 'Blood Sugar'],
    'Value': [age, heart_rate, blood_sugar]
})
st.bar_chart(health_data.set_index('Parameter'))

# Predict button
if st.button('🔍 Predict Heart Attack Risk'):
    input_data = np.array([[age, heart_rate, blood_sugar]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('⚠️ High Risk of Heart Attack! Please consult a doctor.')
        st.markdown("😟 Stay Safe!")
    else:
        st.success('✅ Low Risk of Heart Attack. Stay Healthy!')
        st.markdown("😄 Great Health!")

    st.write('---')

    # Randomly plot a fake "Model Accuracy" graph
    st.subheader('📈 Model Performance Visualization')
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10), marker='o', linestyle='-')
    ax.set_title('Model Random Accuracy Trend')
    st.pyplot(fig)

st.write('---')
st.caption('Developed with ❤️ by [Your Name]')
