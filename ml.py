# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset/data.csv')

# Print the initial shape of the dataset
print(f'Initial dataset shape: {data.shape}')

# Check for empty values
print(data.isnull().sum())

# Drop the 'Unnamed: 32' column if it exists
data.drop(columns=['Unnamed: 32'], inplace=True, errors='ignore')

# Clean the dataset by dropping rows with empty values
data.dropna(inplace=True)

# Check the shape after cleaning
if data.shape[0] == 0:
    raise ValueError("All samples have been removed due to empty values.")

# Print the shape after cleaning
print(f'Dataset shape after cleaning: {data.shape}')

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
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the results
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosis', data=data)
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis (0: Benign, 1: Malignant)')
plt.ylabel('Count')
plt.show()