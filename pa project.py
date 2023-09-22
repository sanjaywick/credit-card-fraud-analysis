import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

df = pd.read_csv("dataset1.csv")

df = df.dropna()

print(df.describe())



### Separate the features from the target variable
##columns = [i for i in df.columns if i not in ['Time (in days)', 'Amount', 'class']]
##X = df[columns]
##
### Standardize the features
##scaler = StandardScaler()
##X_scaled = scaler.fit_transform(X)
##
### Perform PCA
##pca = PCA()
##pca.fit(X_scaled)
##
### Calculate explained variance ratio
##explained_variance = pca.explained_variance_ratio_
##
### Plot the explained variance ratio
##plt.figure(figsize=(10, 6))
##plt.bar(range(len(columns)), explained_variance)
##plt.xticks(range(len(columns)), columns, rotation=90)
##plt.xlabel('Principal Components')
##plt.ylabel('Explained Variance Ratio')
##plt.title('Explained Variance Ratio for Principal Components')
##plt.tight_layout()
##plt.show()

# Separate the features from the target variable
columns = [ i for i in df.columns if i not in ['Time (in days)' , 'Amount' , 'class']]
print(columns)
X = df[columns]
y=df['class']
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

# Apply PCA to the scaled features
X_pca = pca.fit_transform(X_scaled)
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
for i in range(len(columns)):
    print("Column ", columns[i],":-")
    print("Explained Variance Ratio:", explained_variance[i])
    print()

import seaborn as sns

num_components = X_pca.shape[1]
column_names = [f"PC{i+1}" for i in range(num_components)]
X_pca_df = pd.DataFrame(X_pca, columns=column_names)

# Visualize the correlation heatmap
plt.figure(figsize = (100,100))
sns.heatmap(X_pca_df.corr().round(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Principal Components')
plt.show()

# Assuming 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize the classifier
classifier = SVC()

# Train the classifier on the training data
classifier.fit(X_train, y_train)
# Use the trained classifier to make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Classification:\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

silhouette_avg = silhouette_score(X_pca, y)

print("Silhouette Score:", silhouette_avg)

