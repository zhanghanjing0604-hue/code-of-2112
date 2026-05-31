import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def evaluate_predictions(model_name, y_true, y_pred, y_prob=None, pos_label=None):
    print(f"\n===== {model_name} Evaluation =====")

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    if y_prob is not None and pos_label is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        print(f"{model_name} AUC:", roc_auc)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"{model_name} ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.show()


df = pd.read_excel("global_bleaching_environmental.xlsx")

target_col = "Bleaching_Level"

features = [
    "Latitude_Degrees",
    "Longitude_Degrees",
    "Distance_to_Shore",
    "Depth_m",
    "Percent_Cover",
    "Temperature_Mean",
    "Temperature_Maximum",
    "SSTA",
    "SSTA_DHW",
    "TSA",
    "TSA_DHWMean"
]

df = df.dropna(subset=[target_col])

X = df[features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
y = df[target_col].astype(str)

print("\nClass distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

gnb = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", GaussianNB())
])

log_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

models = {
    "Gaussian Naive Bayes": gnb,
    "Logistic Regression": log_model,
    "Random Forest": rf_model
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n===== 5-fold Cross Validation =====")
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_weighted"
    )
    print(f"{name} CV F1 scores:", scores)
    print(f"{name} mean CV F1:", scores.mean())

print("\n===== Train/Test Evaluation =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = None
    pos_label = None

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "Population" in classes:
            pos_label = "Population"
            positive_index = classes.index(pos_label)
            y_prob = model.predict_proba(X_test)[:, positive_index]

    evaluate_predictions(
        model_name=name,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        pos_label=pos_label
    )