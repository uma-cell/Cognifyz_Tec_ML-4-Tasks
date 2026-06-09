import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("dataset.csv")
print("\nDataset Loaded:", df.shape)

# ---------------------------
# 2. Clean dataset
# ---------------------------
df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# ---------------------------
# 3. Target
# ---------------------------
target = "Cuisines"

# ---------------------------
# 4. Encoding
# ---------------------------
le_dict = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# ---------------------------
# 5. Split data
# ---------------------------
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 6. Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------
# 7. Accuracy
# ---------------------------
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print("📊 MODEL ACCURACY")
print("==============================")
print("Accuracy:", round(acc, 2))

# ---------------------------
# 8. SAMPLE PREDICTIONS (CLEAN FORMAT)
# ---------------------------
print("\n==============================")
print("🍽️ SAMPLE PREDICTIONS")
print("==============================")

sample_data = X_test.sample(5, random_state=42)
sample_pred = model.predict(sample_data)

decoded = le_dict['Cuisines'].inverse_transform(sample_pred)

for i in range(5):
    print("\n----------------------------------")
    print(f"Sample {i+1}")
    print("----------------------------------")
    
    print("\n📥 Input Features:")
    
    # clean feature printing (not messy dict)
    for k, v in sample_data.iloc[i].items():
        print(f"   {k}: {v}")

    print("\n🍽️ Predicted Cuisine:")
    print(f"👉 {decoded[i]}")
    
    print("----------------------------------")