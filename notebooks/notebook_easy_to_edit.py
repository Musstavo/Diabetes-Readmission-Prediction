# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import run_pipeline
from src.model import get_xgb_model
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


# %%
data = pd.read_csv("../data/diabetic_data.csv")

X = run_pipeline(data)

y = data["readmitted"].replace({"<30": 1, ">30": 0, "NO": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# %%
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = get_xgb_model(scale_weight)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)


# %%
model.save_model("../src/diabetes_xgb_model.json")
print("Production model saved to src/ folder.")

# %%
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Top 10 Risk Factors for Readmission")
plt.show()
