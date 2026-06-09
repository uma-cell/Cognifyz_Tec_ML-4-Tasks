# =========================================
# RESTAURANT RATING PREDICTION PROJECT
# =========================================

# IMPORT REQUIRED LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("Dataset.csv")


# =========================================
# HANDLE MISSING VALUES
# =========================================

df['Cuisines'] = df['Cuisines'].fillna('Unknown')


# =========================================
# DROP UNNECESSARY COLUMNS
# =========================================

columns_to_drop = [
    'Restaurant ID',
    'Restaurant Name',
    'Address',
    'Locality',
    'Locality Verbose',
    'Rating color',
    'Rating text'
]

df = df.drop(columns=columns_to_drop, errors='ignore')


# =========================================
# ENCODE CATEGORICAL COLUMNS
# =========================================

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column].astype(str))


# =========================================
# DEFINE FEATURES AND TARGET
# =========================================

X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']


# =========================================
# SPLIT TRAINING AND TESTING DATA
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================
# LINEAR REGRESSION MODEL
# =========================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)


# =========================================
# DECISION TREE MODEL
# =========================================

decision_model = DecisionTreeRegressor(random_state=42)

decision_model.fit(X_train, y_train)

decision_predictions = decision_model.predict(X_test)

decision_mse = mean_squared_error(y_test, decision_predictions)
decision_r2 = r2_score(y_test, decision_predictions)


# =========================================
# PRINT RESULTS
# =========================================

print("\n")
print("========== LINEAR REGRESSION RESULTS ==========")
print(f"Mean Squared Error : {linear_mse:.2f}")
print(f"R2 Score           : {linear_r2:.2f}")


print("\n")
print("========== DECISION TREE RESULTS ==========")
print(f"Mean Squared Error : {decision_mse:.2f}")
print(f"R2 Score           : {decision_r2:.2f}")


# =========================================
# BEST MODEL
# =========================================

print("\n")

if decision_r2 > linear_r2:
    print("Best Model : Decision Tree Regressor")
else:
    print("Best Model : Linear Regression")


# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = decision_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n")
print("========== TOP IMPORTANT FEATURES ==========")

print(feature_importance.head(10))


# =========================================
# SAMPLE PREDICTIONS
# =========================================

comparison = pd.DataFrame({
    'Actual Rating': y_test.values,
    'Predicted Rating': decision_predictions
})

print("\n")
print("========== SAMPLE PREDICTIONS ==========")

print(comparison.head(10))


# =========================================
# SAVE PREDICTIONS TO CSV
# =========================================

comparison.to_csv("Predicted_Ratings.csv", index=False)

print("\n")
print("Predicted_Ratings.csv file saved successfully!")


# =========================================
# FEATURE IMPORTANCE GRAPH
# =========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top Important Features")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.show()


# =========================================
# ACTUAL VS PREDICTED GRAPH
# =========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, decision_predictions)

plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")

plt.title("Actual vs Predicted Ratings")

plt.tight_layout()

plt.show()


# =========================================
# END OF PROJECT
# =========================================