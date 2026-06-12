import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("german_credit_data.csv", index_col=0)

print("Columns in dataset:")
print(df.columns)

# Fill missing values safely
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace("nan", pd.NA)
        df[col] = df[col].fillna(df[col].mode()[0])

# Create target column if missing
if "Risk" not in df.columns:
    # Simple rule: smaller credit amount = lower risk
    median_credit = df["Credit amount"].median()
    df["Risk"] = (df["Credit amount"] > median_credit).astype(int)

# Encode categorical columns
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

best_model = None
best_auc = 0
best_model_name = ""

# Train models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name}")
print(f"Best ROC-AUC: {best_auc:.4f}")

# Confusion Matrix
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
joblib.dump(best_model, "credit_score_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")

# Sample prediction
sample = X.iloc[0:1]
sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)[0]
probability = best_model.predict_proba(sample_scaled)[0][1]

print("\nPrediction:", "Good Credit" if prediction == 0 else "Bad Credit")
print("Probability:", probability)