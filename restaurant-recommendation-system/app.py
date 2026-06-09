# ============================================
# RESTAURANT RECOMMENDATION SYSTEM
# ============================================

# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# STEP 1 : LOAD DATASET
# ============================================

# IMPORTANT:
# Your CSV file name should be Dataset.csv
# If different name, change below line

df = pd.read_csv("Dataset.csv")

# ============================================
# STEP 2 : HANDLE MISSING VALUES
# ============================================

df['Cuisines'] = df['Cuisines'].fillna('Unknown')

df['Price range'] = df['Price range'].fillna(1)

# ============================================
# STEP 3 : SELECT IMPORTANT COLUMNS
# ============================================

data = df[['Restaurant Name',
           'Cuisines',
           'Price range',
           'Aggregate rating']]

# ============================================
# STEP 4 : CREATE COMBINED FEATURES
# ============================================

data['combined_features'] = (
    data['Cuisines'] + " " +
    data['Price range'].astype(str)
)

# ============================================
# STEP 5 : TEXT TO NUMBERS
# ============================================

cv = CountVectorizer()

count_matrix = cv.fit_transform(data['combined_features'])

# ============================================
# STEP 6 : CALCULATE SIMILARITY
# ============================================

similarity = cosine_similarity(count_matrix)

# ============================================
# STEP 7 : RECOMMENDATION FUNCTION
# ============================================

def recommend_restaurants(cuisine, price_range):

    # User input combine
    user_input = cuisine + " " + str(price_range)

    # Convert user input into vector
    user_vector = cv.transform([user_input])

    # Similarity scores
    scores = cosine_similarity(user_vector, count_matrix)

    # Add similarity scores
    data['Similarity Score'] = scores[0]

    # Sort recommendations
    recommendations = data.sort_values(
        by=['Similarity Score', 'Aggregate rating'],
        ascending=False
    )

    # Remove duplicates
    recommendations = recommendations.drop_duplicates(
        subset='Restaurant Name'
    )

    # Top 5 restaurants
    return recommendations[
        ['Restaurant Name',
         'Cuisines',
         'Price range',
         'Aggregate rating']
    ].head(5)

# ============================================
# STEP 8 : USER INPUT
# ============================================

print("================================")
print("RESTAURANT RECOMMENDATION SYSTEM")
print("================================")

# User enters preferences
user_cuisine = input("\nEnter Cuisine Type: ")

user_price = int(input("Enter Price Range (1-4): "))

# ============================================
# STEP 9 : GET RECOMMENDATIONS
# ============================================

result = recommend_restaurants(
    user_cuisine,
    user_price
)

# ============================================
# STEP 10 : DISPLAY OUTPUT
# ============================================

print("\n================================")
print("USER PREFERENCES")
print("================================")

print("Cuisine Type :", user_cuisine)

print("Price Range  :", user_price)

print("\n================================")
print("RECOMMENDED RESTAURANTS")
print("================================")

for index, row in result.iterrows():

    print("\nRestaurant Name :", row['Restaurant Name'])

    print("Cuisine         :", row['Cuisines'])

    print("Price Range     :", row['Price range'])

    print("Rating          :", row['Aggregate rating'])

# ============================================
# END
# ============================================