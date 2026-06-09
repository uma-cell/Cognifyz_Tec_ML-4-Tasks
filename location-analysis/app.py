import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("dataset.csv")

print("\nDataset Shape:", df.shape)

# clean column spaces
df.columns = df.columns.str.strip()

# =========================
# 2. CLEAN LAT/LONG DATA
# =========================
df = df.dropna(subset=['Latitude', 'Longitude'])

# =========================
# 3. CREATE MAP
# =========================

# center map based on dataset
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]

restaurant_map = folium.Map(location=map_center, zoom_start=11)

# =========================
# 4. ADD RESTAURANT MARKERS
# =========================
for _, row in df.iterrows():

    popup_text = f"""
    🍽 Restaurant Name: {row.get('Restaurant Name', 'Unknown')} <br>
    📍 Location: Restaurant Location <br>
    ⭐ Rating: {row.get('Aggregate rating', 'N/A')}
    """

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(restaurant_map)

# save map
restaurant_map.save("restaurants_map.html")

print("\n✅ Map created successfully!")
print("👉 Open 'restaurants_map.html' in browser")

# =========================
# 5. GROUP BY CITY / LOCALITY
# =========================
group_col = 'City' if 'City' in df.columns else 'Locality'

city_counts = df[group_col].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=city_counts.values, y=city_counts.index, palette="viridis")
plt.title("Top Areas with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City/Locality")
plt.tight_layout()
plt.show()

# =========================
# 6. AVERAGE RATING BY CITY
# =========================
if 'Aggregate rating' in df.columns:
    rating_city = df.groupby(group_col)['Aggregate rating'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    sns.barplot(x=rating_city.values, y=rating_city.index, palette="coolwarm")
    plt.title("Average Rating by Area")
    plt.xlabel("Average Rating")
    plt.ylabel(group_col)
    plt.tight_layout()
    plt.show()

# =========================
# 7. PRICE ANALYSIS
# =========================
if 'Price range' in df.columns:
    price_city = df.groupby(group_col)['Price range'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    sns.barplot(x=price_city.values, y=price_city.index, palette="magma")
    plt.title("Price Range by Area")
    plt.xlabel("Price Range")
    plt.ylabel(group_col)
    plt.tight_layout()
    plt.show()

# =========================
# 8. INSIGHTS
# =========================
print("\n===== INSIGHTS =====")

print(f"📍 Most restaurants are in: {city_counts.index[0]}")

if 'Aggregate rating' in df.columns:
    print(f"⭐ Highest rated area: {rating_city.index[0]}")

if 'Price range' in df.columns:
    print(f"💰 Most expensive area: {price_city.index[0]}")

print("\n✅ Analysis Completed Successfully!")