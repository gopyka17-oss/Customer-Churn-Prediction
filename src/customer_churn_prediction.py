# ============================================================
# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING
# Keep Churn_Modelling.csv in the same folder as this file.
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False


def save_show_plot(filename):
    os.makedirs("images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{filename}", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_model(model_name, model, X_test_data, y_test, results):
    y_pred = model.predict(X_test_data)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_data)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan
    })

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_show_plot(f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")

    return y_prob


print("Loading dataset...")

df = pd.read_csv("C:/Users/gopik/OneDrive/Desktop/Abseena miss/datasets/Churn_Modelling.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

print("\nColumns after removing unnecessary identifiers:")
print(df.columns)


# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n==============================")
print("EXPLORATORY DATA ANALYSIS")
print("==============================")

# ------------------------------------------------------------
# Target Variable Distribution
# ------------------------------------------------------------

print("\nCustomer Churn Distribution")
print(df["Exited"].value_counts())

print("\nCustomer Churn Percentage")
print((df["Exited"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Exited", palette="Set2")
plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
save_show_plot("01_churn_distribution.png")


# ------------------------------------------------------------
# Geography Distribution
# ------------------------------------------------------------

print("\nCustomers by Geography")
print(df["Geography"].value_counts())

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="Geography", palette="viridis")
plt.title("Customers by Geography")
save_show_plot("02_geography_distribution.png")


# ------------------------------------------------------------
# Gender Distribution
# ------------------------------------------------------------

print("\nCustomers by Gender")
print(df["Gender"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", palette="pastel")
plt.title("Customers by Gender")
save_show_plot("03_gender_distribution.png")


# ------------------------------------------------------------
# Churn by Geography
# ------------------------------------------------------------

geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)

print("\nAverage Churn Rate by Geography")
print(geo_churn)

plt.figure(figsize=(7,5))
sns.barplot(x=geo_churn.index, y=geo_churn.values, palette="rocket")
plt.title("Average Churn Rate by Geography")
plt.ylabel("Churn Rate")
save_show_plot("04_churn_geography.png")


# ------------------------------------------------------------
# Churn by Gender
# ------------------------------------------------------------

gender_churn = df.groupby("Gender")["Exited"].mean()

print("\nAverage Churn Rate by Gender")
print(gender_churn)

plt.figure(figsize=(6,4))
sns.barplot(x=gender_churn.index, y=gender_churn.values, palette="coolwarm")
plt.title("Average Churn Rate by Gender")
plt.ylabel("Churn Rate")
save_show_plot("05_churn_gender.png")


# ------------------------------------------------------------
# Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution")
save_show_plot("06_age_distribution.png")


# ------------------------------------------------------------
# Age vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(data=df,
             x="Age",
             hue="Exited",
             bins=30,
             kde=True,
             palette="Set1")

plt.title("Age Distribution by Churn")
save_show_plot("07_age_vs_churn.png")


# ------------------------------------------------------------
# Credit Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["CreditScore"], bins=30, kde=True)
plt.title("Credit Score Distribution")
save_show_plot("08_credit_score_distribution.png")


# ------------------------------------------------------------
# Balance Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Balance"], bins=30, kde=True)
plt.title("Balance Distribution")
save_show_plot("09_balance_distribution.png")


# ------------------------------------------------------------
# Salary Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["EstimatedSalary"], bins=30, kde=True)
plt.title("Estimated Salary Distribution")
save_show_plot("10_salary_distribution.png")


# ------------------------------------------------------------
# Boxplots
# ------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(data=df,
            x="Exited",
            y="Age",
            palette="Set3")

plt.title("Age vs Churn")
save_show_plot("11_age_boxplot.png")


plt.figure(figsize=(6,5))
sns.boxplot(data=df,
            x="Exited",
            y="Balance",
            palette="Set2")

plt.title("Balance vs Churn")
save_show_plot("12_balance_boxplot.png")


# ------------------------------------------------------------
# Active Member vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(6,4))
sns.barplot(data=df,
            x="IsActiveMember",
            y="Exited",
            palette="viridis")

plt.title("Active Member vs Churn")
save_show_plot("13_active_member_churn.png")


# ------------------------------------------------------------
# Number of Products vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(7,4))
sns.barplot(data=df,
            x="NumOfProducts",
            y="Exited",
            palette="crest")

plt.title("Products vs Churn")
save_show_plot("14_products_churn.png")


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

df_corr = df.copy()

df_corr["Gender"] = df_corr["Gender"].map({
    "Male":0,
    "Female":1
})

df_corr = pd.get_dummies(
    df_corr,
    columns=["Geography"],
    drop_first=True,
    dtype=int
)

plt.figure(figsize=(14,10))

sns.heatmap(df_corr.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
save_show_plot("15_correlation_heatmap.png")


print("\nEDA Completed Successfully.")


# ============================================================
# FEATURE ENGINEERING & DATA PREPROCESSING
# ============================================================

print("\n==============================")
print("FEATURE ENGINEERING")
print("==============================")

# ------------------------------------------------------------
# Create Age Groups
# ------------------------------------------------------------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18,30,40,50,60,100],
    labels=[
        "18-30",
        "31-40",
        "41-50",
        "51-60",
        "60+"
    ]
)

print(df["AgeGroup"].value_counts())

plt.figure(figsize=(8,5))
sns.barplot(
    data=df,
    x="AgeGroup",
    y="Exited",
    palette="magma"
)

plt.title("Customer Churn by Age Group")
plt.ylabel("Average Churn")
save_show_plot("16_age_group_churn.png")


# ------------------------------------------------------------
# Create Balance Category
# ------------------------------------------------------------

df["BalanceCategory"] = pd.cut(
    df["Balance"],
    bins=[-1, 0, 50000, 100000, 300000],
    labels=[
        "Zero Balance",
        "Low",
        "Medium",
        "High"
    ]
)
print(df["BalanceCategory"].value_counts())


plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="BalanceCategory",
    y="Exited",
    palette="viridis"
)

plt.title("Balance Category vs Churn")

save_show_plot("17_balance_category.png")


# ------------------------------------------------------------
# Estimated Salary Category
# ------------------------------------------------------------

df["SalaryCategory"] = pd.qcut(
    df["EstimatedSalary"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="SalaryCategory",
    y="Exited",
    palette="plasma"
)

plt.title("Salary Category vs Churn")

save_show_plot("18_salary_category.png")


# ------------------------------------------------------------
# Encoding
# ------------------------------------------------------------

print("\nEncoding categorical variables...")

df_model = pd.get_dummies(
    df,
    columns=[
        "Geography",
        "Gender",
        "AgeGroup",
        "BalanceCategory",
        "SalaryCategory"
    ],
    drop_first=True,
    dtype=int
)

print(df_model.head())


# ------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------

X = df_model.drop("Exited",axis=1)

y = df_model["Exited"]

print("\nFeature Matrix Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)


# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train,X_test,y_train,y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)


# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

print("\nScaling Numerical Variables...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("Scaling Completed")


# ------------------------------------------------------------
# Class Distribution
# ------------------------------------------------------------

print("\nOriginal Class Distribution")

print(y_train.value_counts())


# ------------------------------------------------------------
# Handle Class Imbalance
# ------------------------------------------------------------

if SMOTE_AVAILABLE:

    print("\nApplying SMOTE...")

    smote = SMOTE(random_state=42)

    X_train_smote,y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    X_train_scaled_smote,y_train_scaled_smote = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    print("\nBefore SMOTE")

    print(y_train.value_counts())

    print("\nAfter SMOTE")

    print(pd.Series(y_train_smote).value_counts())

else:

    print("\nSMOTE not installed")

    print("Run")

    print("pip install imbalanced-learn")

    X_train_smote = X_train

    y_train_smote = y_train

    X_train_scaled_smote = X_train_scaled

    y_train_scaled_smote = y_train


# ------------------------------------------------------------
# Save Processed Dataset
# ------------------------------------------------------------

os.makedirs("processed_data",exist_ok=True)

df_model.to_csv(
    "processed_data/processed_churn_dataset.csv",
    index=False
)

print("\nProcessed dataset saved.")


# ------------------------------------------------------------
# Ready For Machine Learning
# ------------------------------------------------------------

print("\n===================================")

print("Feature Engineering Completed")

print("Dataset Ready For Machine Learning")

print("===================================")

# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

print("\n==============================")
print("MODEL TRAINING")
print("==============================")

model_results = []

# ------------------------------------------------------------
# 1. Logistic Regression
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_scaled_smote, y_train_scaled_smote)

log_prob = evaluate_model(
    "Logistic Regression",
    log_model,
    X_test_scaled,
    y_test,
    model_results
)


# ------------------------------------------------------------
# 2. Decision Tree
# ------------------------------------------------------------

print("\nTraining Decision Tree...")

dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

dt_model.fit(X_train_smote, y_train_smote)

dt_prob = evaluate_model(
    "Decision Tree",
    dt_model,
    X_test,
    y_test,
    model_results
)


# ------------------------------------------------------------
# 3. Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_smote, y_train_smote)

rf_prob = evaluate_model(
    "Random Forest",
    rf_model,
    X_test,
    y_test,
    model_results
)


# ------------------------------------------------------------
# 4. XGBoost
# ------------------------------------------------------------

if XGBOOST_AVAILABLE:

    print("\nTraining XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train_smote, y_train_smote)

    xgb_prob = evaluate_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test,
        model_results
    )

else:

    print("\nXGBoost is not installed.")
    print("To install XGBoost, run:")
    print("pip install xgboost")

    xgb_model = None
    xgb_prob = None


# ------------------------------------------------------------
# Convert Results to DataFrame
# ------------------------------------------------------------

results_df = pd.DataFrame(model_results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON TABLE")
print("==============================")

print(results_df)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


# ------------------------------------------------------------
# Model Comparison Chart
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score",
    palette="Set2"
)

plt.title("Model Comparison Based on F1 Score")
plt.xticks(rotation=30)

save_show_plot("19_model_comparison_f1.png")


# ------------------------------------------------------------
# Accuracy Comparison
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy",
    palette="viridis"
)

plt.title("Model Comparison Based on Accuracy")
plt.xticks(rotation=30)

save_show_plot("20_model_comparison_accuracy.png")


# ------------------------------------------------------------
# ROC Curve Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

if log_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, log_prob)
    plt.plot(fpr, tpr, label="Logistic Regression")

if dt_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, dt_prob)
    plt.plot(fpr, tpr, label="Decision Tree")

if rf_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, rf_prob)
    plt.plot(fpr, tpr, label="Random Forest")

if xgb_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, xgb_prob)
    plt.plot(fpr, tpr, label="XGBoost")

plt.plot([0,1], [0,1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

save_show_plot("21_roc_curve_comparison.png")


# ------------------------------------------------------------
# Feature Importance - Random Forest
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("TOP IMPORTANT FEATURES")
print("==============================")

print(feature_importance.head(15))

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
    palette="crest"
)

plt.title("Top 10 Important Features - Random Forest")

save_show_plot("22_feature_importance.png")


print("\nModel training and evaluation completed successfully.")


# ============================================================
# HYPERPARAMETER TUNING, FINAL MODEL, INSIGHTS
# ============================================================

print("\nRunning hyperparameter tuning for Random Forest...")

param_grid = {
    "n_estimators": [100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=1
)

grid_search.fit(X_train_smote, y_train_smote)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(grid_search.best_score_)

best_rf_model = grid_search.best_estimator_

tuned_rf_prob = evaluate_model(
    "Tuned Random Forest",
    best_rf_model,
    X_test,
    y_test,
    model_results
)

final_results_df = pd.DataFrame(model_results)

final_results_df = final_results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFINAL MODEL COMPARISON:")
print(final_results_df)

final_results_df.to_csv(
    "final_model_comparison_results.csv",
    index=False
)

# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(best_rf_model, "models/best_random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully inside models folder.")

# ------------------------------------------------------------
# Final ROC curve for tuned model
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

fpr, tpr, _ = roc_curve(y_test, tuned_rf_prob)

plt.plot(fpr, tpr, label="Tuned Random Forest")
plt.plot([0,1], [0,1], linestyle="--")

plt.title("ROC Curve - Tuned Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

save_show_plot("23_tuned_random_forest_roc.png")

# ------------------------------------------------------------
# Business insights
# ------------------------------------------------------------

print("""
BUSINESS INSIGHTS:

1. Age is one of the strongest indicators of churn.
   Older customers show a higher probability of leaving the bank.

2. Geography influences churn behaviour.
   Some regions show higher churn rates, so location-based retention
   strategies may be useful.

3. Active membership reduces churn.
   Inactive customers are more likely to leave, so banks should improve
   engagement through offers, reminders, and loyalty benefits.

4. Number of products matters.
   Customers with fewer products may be less attached to the bank.

5. Balance and salary-related patterns can help identify high-value
   customers who may need personalised retention campaigns.

6. Random Forest and XGBoost are useful because they capture non-linear
   patterns better than simple models.
""")

# ------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------

print("""
CONCLUSION:

This project developed a complete machine learning pipeline for
customer churn prediction.

The workflow included:
- Data loading
- Data cleaning
- Exploratory data analysis
- Feature engineering
- Data preprocessing
- Handling class imbalance using SMOTE
- Machine learning model building
- Model comparison
- Hyperparameter tuning
- Feature importance analysis
- Business insight generation

The final model can help a bank identify customers who are likely to churn.
This allows the business to take preventive action through personalised
offers, customer engagement campaigns, loyalty benefits, and improved
customer support.

This project is suitable for a GitHub portfolio because it demonstrates
end-to-end data science skills from raw data to business decision support.
""")

print("\nPROJECT COMPLETED SUCCESSFULLY.")
print("Check these folders/files:")
print("- images/ : contains saved charts")
print("- models/ : contains saved trained model")
print("- processed_data/ : contains processed dataset")
print("- model_comparison_results.csv")
print("- final_model_comparison_results.csv")
print("- feature_importance.csv")

