# app.py

# --------------------------
# Import Libraries
# --------------------------
import streamlit as st
import numpy as np
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
st.title('❤️ Heart Attack Risk Prediction')

st.write("""
## Enter your health information below:
""")

# Take input from user
age = st.number_input('Enter your Age:', min_value=1, max_value=120)
heart_rate = st.number_input('Enter your Heart Rate:', min_value=30, max_value=220)
blood_sugar = st.number_input('Enter your Blood Sugar Level:', min_value=50, max_value=300)

# Predict button
if st.button('Predict Heart Attack Risk'):
    input_data = np.array([[age, heart_rate, blood_sugar]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('⚠️ High Risk of Heart Attack! Please consult a doctor.')
    else:
        st.success('✅ Low Risk of Heart Attack. Stay Healthy!')

st.write('---')
st.caption('Developed with ❤️ by [Your Name]')
