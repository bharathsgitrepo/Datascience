#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install streamlit


# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="KNN Iris App", layout="centered")

st.title("🌸 Iris Classification using KNN")

# Sidebar: Set K
k = st.sidebar.slider("Choose K (number of neighbors)", 1, 15, 5)

# Load data
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
accuracy = knn.score(X_test, y_test)
st.write(f"✅ **Accuracy**: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm, cmap='viridis')
ax.set_title("Confusion Matrix")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 1 else "black")
st.pyplot(fig)

# Classification Report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap='BuGn'))

