import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("heart.csv")  # change filename if needed

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# 2. Correlation Matrix
correlation_matrix = df.corr()

# 3. Data Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=df["target"])
plt.title("Number of Patients with and without Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("countplot_target.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["target"], y=df["age"])
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("age_vs_target.png")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 4. Split Features and Target
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Standardization (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
log_reg_cm = confusion_matrix(y_test, y_pred_log_reg)
log_reg_report = classification_report(y_test, y_pred_log_reg)

print("\nLogistic Regression Accuracy:", log_reg_acc)
print("Confusion Matrix:\n", log_reg_cm)
print("Classification Report:\n", log_reg_report)

# 7. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)
dt_cm = confusion_matrix(y_test, y_pred_dt)
dt_report = classification_report(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_acc)
print("Confusion Matrix:\n", dt_cm)
print("Classification Report:\n", dt_report)

# 8. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_cm = confusion_matrix(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", rf_acc)
print("Confusion Matrix:\n", rf_cm)
print("Classification Report:\n", rf_report)

# 9. Precision, Recall, F1-Score Comparison
log_reg_scores = precision_recall_fscore_support(y_test, y_pred_log_reg, average='binary')
dt_scores = precision_recall_fscore_support(y_test, y_pred_dt, average='binary')
rf_scores = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')

print("\nModel Comparison:")
print("Logistic Regression - Precision:", log_reg_scores[0], "Recall:", log_reg_scores[1], "F1-score:", log_reg_scores[2])
print("Decision Tree       - Precision:", dt_scores[0], "Recall:", dt_scores[1], "F1-score:", dt_scores[2])
print("Random Forest       - Precision:", rf_scores[0], "Recall:", rf_scores[1], "F1-score:", rf_scores[2])

# 10. Best Model Selection
best_model = max([
    (log_reg_acc, "Logistic Regression"),
    (dt_acc, "Decision Tree"),
    (rf_acc, "Random Forest")
])

print(f"\nBest Model: {best_model[1]} with Accuracy {best_model[0]}")

# 11. Confusion Matrix Heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, model, cm in zip(
    axes,
    ["Logistic Regression", "Decision Tree", "Random Forest"],
    [log_reg_cm, dt_cm, rf_cm]
):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

print("\nProcess completed successfully.")
