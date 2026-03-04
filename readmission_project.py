# Databricks notebook source
# Basic data handling
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import roc_auc_score

# Visualization (optional but useful later)
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC This block imports the libraries needed for the project. Pandas and NumPy are used for data manipulation, while scikit-learn provides tools for splitting the dataset, training machine learning models, and evaluating performance. I also imported visualization libraries like Matplotlib and Seaborn for plotting results later, and SHAP for interpreting the model’s predictions.

# COMMAND ----------

# Load the modeling view we created in SQL
# Spark will read the view directly

df = spark.sql("SELECT * FROM diabetes_model_table")

# Convert to pandas for scikit-learn modeling
pdf = df.toPandas()

print("Rows:", pdf.shape[0])
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In this step, I load the diabetes_model_table view from Databricks into a Spark dataframe and then convert it into a Pandas dataframe. Spark is great for working with large datasets, but scikit-learn models expect data in a Pandas format. Converting it allows me to use the standard Python machine learning ecosystem.

# COMMAND ----------

# Separate features and target

target = "readmit_30d"

X = pdf.drop(columns=["readmit_30d","encounter_id","patient_nbr"])
y = pdf[target]

# Convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)

print("Feature matrix shape:", X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Here I separate the dataset into the target variable (readmit_30d) and the feature variables used for prediction. I also remove identifiers like encounter_id and patient_nbr since they don’t provide predictive value. For categorical variables such as gender or race, I convert them into numerical dummy variables using one-hot encoding so the machine learning models can process them properly.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC This block splits the data into training and testing sets using an 80/20 ratio. The model is trained on the training set and evaluated on the test set to measure how well it generalizes to unseen data. I also use stratification to preserve the original readmission rate distribution in both sets.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

log_model = LogisticRegression(max_iter=2000)

log_model.fit(X_train, y_train)

log_probs = log_model.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test, log_probs)

print("Logistic Regression ROC-AUC:", round(roc,3))

# COMMAND ----------

# MAGIC %md
# MAGIC Here I train a Logistic Regression model, which is a simple and commonly used baseline model for binary classification problems like predicting readmission. After fitting the model on the training data, I generate predicted probabilities for the test set and calculate the ROC-AUC score to measure how well the model separates high-risk and low-risk patients.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test, rf_probs)

print("Random Forest ROC-AUC:", round(roc,3))

# COMMAND ----------

# MAGIC %md
# MAGIC In this block I train a Random Forest classifier, which is an ensemble model that builds many decision trees and combines their predictions. Random Forest models can capture more complex patterns in the data than logistic regression. After training the model, I again calculate the ROC-AUC score to compare its performance against the baseline model.

# COMMAND ----------


# Rank patients by predicted risk
df_eval = pd.DataFrame({
    "y_true": y_test,
    "risk": rf_probs
})

df_eval = df_eval.sort_values("risk", ascending=False)

top_10_percent = int(len(df_eval) * 0.10)

top = df_eval.head(top_10_percent)

recall_top10 = top["y_true"].sum() / df_eval["y_true"].sum()

print("Recall@Top10%:", round(recall_top10,3))

# COMMAND ----------

# MAGIC %md
# MAGIC This section calculates a business-focused metric called Recall@Top10%, which measures how many true readmissions appear within the top 10% of patients ranked by predicted risk. In practice, hospitals often focus on the highest-risk patients for early intervention programs, so this metric reflects how useful the model would be in a real operational setting.

# COMMAND ----------

# MAGIC %md
# MAGIC Interpreting the Results:
# MAGIC
# MAGIC The Logistic Regression model achieved an ROC-AUC of 0.624, while the Random Forest performed slightly better with 0.637, indicating modest predictive ability. The Recall@Top10% of 0.205 means that roughly 20% of all readmissions are captured within the top 10% highest-risk patients identified by the model. While the dataset has limited clinical detail, the results still demonstrate how machine learning can help prioritize patients who may benefit from early intervention.

# COMMAND ----------

# Get feature importance from the trained Random Forest model

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

# Sort by most important features
importance = importance.sort_values("importance", ascending=False)

importance.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC This block extracts feature importance scores from the Random Forest model. Random Forests naturally track how much each feature contributes to reducing prediction error across the decision trees. Sorting these values allows us to identify which patient characteristics have the strongest influence on readmission risk.

# COMMAND ----------

import matplotlib.pyplot as plt

top_features = importance.head(10)

plt.figure(figsize=(8,5))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top Features Driving Readmission Risk")
plt.xlabel("Importance Score")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This block visualizes the most important predictors identified by the Random Forest model. Plotting the top features helps make the results easier to interpret and communicate. Instead of only reporting model metrics, this visualization shows which variables—such as hospital stay length or prior admissions—are driving the model’s predictions.

# COMMAND ----------

# Create dataframe with predictions

predictions = pd.DataFrame({
    "readmitted_actual": y_test,
    "predicted_risk": rf_probs
})

# Add risk bucket
predictions["risk_group"] = pd.qcut(predictions["predicted_risk"], 10, labels=False)

predictions.head()

# COMMAND ----------

# MAGIC %md
# MAGIC This block builds a dataset containing the actual readmission outcome and the predicted risk score for each patient in the test set. I also create a risk group variable that divides patients into deciles based on predicted risk. This makes it easy to visualize high-risk segments in Tableau.

# COMMAND ----------

predictions.to_csv("/Workspace/readmission_predictions.csv", index=False)