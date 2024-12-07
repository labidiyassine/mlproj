# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset
try:
    data = pd.read_csv('dataset/data.csv')
except FileNotFoundError:
    st.error("The dataset file was not found. Please check the file path.")
    st.stop()  # Stop the execution of the app if the file is not found
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()  # Stop the execution of the app for any other errors

# Print the initial shape of the dataset
st.write(f'Initial dataset shape: {data.shape}')

# Check for empty values
st.write(data.isnull().sum())

# Drop the 'Unnamed: 32' column if it exists
data.drop(columns=['Unnamed: 32'], inplace=True, errors='ignore')

# Clean the dataset by dropping rows with empty values
data.dropna(inplace=True)

# Check the shape after cleaning
if data.shape[0] == 0:
    raise ValueError("All samples have been removed due to empty values.")

# Print the shape after cleaning
st.write(f'Dataset shape after cleaning: {data.shape}')

# Encode categorical variables (if any)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Define features and target variable
X = data.drop(['id', 'diagnosis'], axis=1)  # Drop non-feature columns
y = data['diagnosis']

# Check if there are enough samples to split
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("No samples available for training and testing.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
st.write(f'Accuracy: {accuracy * 100:.2f}%')

# Create subplots for box plots and histograms
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Box plot for 'radius_mean'
sns.boxplot(x='diagnosis', y='radius_mean', data=data, ax=axs[0, 0])
axs[0, 0].set_title("Diagnosis vs Radius Mean")

# Histogram for 'radius_mean'
sns.histplot(data['radius_mean'], bins=30, kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Distribution of Radius Mean")

# Box plot for 'texture_mean'
sns.boxplot(x='diagnosis', y='texture_mean', data=data, ax=axs[1, 0])
axs[1, 0].set_title("Diagnosis vs Texture Mean")

# Histogram for 'texture_mean'
sns.histplot(data['texture_mean'], bins=30, kde=True, ax=axs[1, 1])
axs[1, 1].set_title("Distribution of Texture Mean")

plt.tight_layout()  # Adjust layout to prevent overlap
st.pyplot(fig)  # Display the figure in Streamlit

# Additional Charts
# Scatter plot for 'radius_mean' vs 'area_mean'
st.subheader("Scatter Plot: Radius Mean vs Area Mean")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='radius_mean', y='area_mean', hue='diagnosis', data=data, palette='Set1')
plt.title("Scatter Plot of Radius Mean vs Area Mean")
plt.xlabel("Radius Mean")
plt.ylabel("Area Mean")
plt.legend(title='Diagnosis', loc='upper right', labels=['Benign', 'Malignant'])
st.pyplot(plt)

# Pair plot for selected features
st.subheader("Pair Plot of Selected Features")
selected_features = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']]
pair_plot_fig = sns.pairplot(selected_features, hue='diagnosis', palette='Set1')
st.pyplot(pair_plot_fig)