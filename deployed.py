import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the trained model
model = LogisticRegression(max_iter=200)
iris = load_iris()
X = iris.data
y = iris.target
model.fit(X, y)

# Streamlit app
st.title('Simple Iris Flower Prediction App')

# Sidebar for user input
st.sidebar.header('Input Features')

# Input feature values
sepal_length = st.sidebar.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Make prediction
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display prediction
species = iris.target_names[prediction[0]]
st.write(f'Predicted Species: {species}')


picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)



# # Step 1: Import necessary libraries
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Step 2: Load the dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Step 3: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Model Training
# model = LogisticRegression(max_iter=200)  # Create a logistic regression model
# model.fit(X_train, y_train)  # Train the model

# # Step 5: Model Evaluation
# y_pred = model.predict(X_test)  # Make predictions on the testing data
# accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
# print("Accuracy:", accuracy)
