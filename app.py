import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("student_performance_dataset.csv")

# Features and Target
X = data[['study_hours',
          'attendance',
          'previous_marks',
          'assignments_completed']]

y = data['final_marks']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🎓 Student Performance Prediction System")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# User Inputs
study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0)
assignments = st.number_input("Assignments Completed", min_value=0.0)

if st.button("Predict Marks"):

    prediction = model.predict(pd.DataFrame([{
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_marks': previous_marks,
        'assignments_completed': assignments
    }]))

    marks = prediction[0]

    if marks >= 90:
        performance = "Excellent"
    elif marks >= 75:
        performance = "Good"
    elif marks >= 60:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    st.success(f"Predicted Marks: {marks:.2f}")

    if performance == "Excellent":
        st.success("🌟 Performance: Excellent")
    elif performance == "Good":
        st.info("👍 Performance: Good")
    elif performance == "Average":
        st.warning("⚠️ Performance: Average")
    else:
        st.error("❌ Performance: Needs Improvement")

    # Graph
    fig, ax = plt.subplots()

    ax.scatter(
        data['study_hours'],
        data['final_marks'],
        label="Dataset"
    )

    ax.scatter(
        study_hours,
        marks,
        s=500,
        marker="*",
        color="red",
        label="Your Prediction"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Marks")
    ax.set_title("Student Marks Prediction Visualization")
    ax.legend()

    st.pyplot(fig)

    st.success(
        f"📍 Your point on graph: ({study_hours}, {marks:.2f})"
    )