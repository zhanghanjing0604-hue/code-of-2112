import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv("Rita_Coral_Clean.csv")
df["Bleaching_Label"] = (df["Percent_Bleaching"] == 0).astype(int)
df = df.drop(columns=["Percent_Bleaching"])

X = df.drop(columns=["Bleaching_Label"])
y = df["Bleaching_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
)


feature_weight_dict = {

    'Temperature_Kelvin': 1.0,
    'SSTA': 10,
    'SSTA_Maximum': 1.0,
    'SSTA_Frequency': 1.0,
    'SSTA_DHW': 1.0,
    
   
    'Cyclone_Frequency': 0.1,   
    'Distance_to_Shore': 0.1,   
    'Depth_m': 0.5,             
    'Turbidity': 0.5,           
    'Windspeed': 0.5           
}

# Map the dictionary to match the exact column order of X
custom_weights = [feature_weight_dict[col] for col in X.columns]
                  


model = XGBClassifier(
    eval_metric="logloss",
    random_state=50,
    reg_lambda=15,          #Strong L2 reg
    max_depth=5,            #shallow tree to avoid random corelation
    learning_rate=0.1
)

model.fit(X_train, y_train, feature_weights=custom_weights)

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance (With Custom L2-Style Suppression)")
plt.bar(range(X.shape[1]), importances[indices], color="teal", align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

print(f"Final AUC Score: {roc_auc:.3f}")