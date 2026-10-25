
# Question:
# Perform visualization on the Cocoa dataset.
# Display dataset characteristics and attribute types.
# Create a line chart of reviews per year, a bar chart of the
# 10 highest-rated locations with at least 20 reviews, a pie chart
# of rating classes, a histogram of Cocoa Percent, and a scatter
# plot of review year versus mean rating per year.

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Cocoa dataset
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")
df.columns = df.columns.str.strip()

# 2. Display dataset characteristics and attribute types
print("===== DATASET CHARACTERISTICS =====")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nFirst 5 records:")
print(df.head())

# 3. Clean and convert required columns
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Review Date"] = pd.to_numeric(df["Review Date"], errors="coerce")

df["Cocoa Percent"] = (
    df["Cocoa Percent"]
    .astype(str)
    .str.replace("%", "", regex=False)
)
df["Cocoa Percent"] = pd.to_numeric(
    df["Cocoa Percent"], errors="coerce"
)

# Keep rows with valid review year and rating for rating-based charts
rating_data = df.dropna(
    subset=["Review Date", "Rating"]
).copy()

rating_data["Review Date"] = rating_data["Review Date"].astype(int)

# 4. Line chart: Number of reviews per year
reviews_per_year = rating_data.groupby("Review Date").size()

plt.figure(figsize=(9, 5))
plt.plot(
    reviews_per_year.index,
    reviews_per_year.values,
    marker="o"
)
plt.title("Number of Cocoa Reviews per Year")
plt.xlabel("Review Year")
plt.ylabel("Number of Reviews")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Bar chart: Top 10 highest-rated company locations
# Only locations with at least 20 valid reviews are included
location_data = rating_data.dropna(
    subset=["Company Location"]
).copy()

location_summary = location_data.groupby("Company Location").agg(
    Mean_Rating=("Rating", "mean"),
    Number_of_Reviews=("Rating", "count")
)

location_summary = location_summary[
    location_summary["Number_of_Reviews"] >= 20
]

top_locations = location_summary.sort_values(
    "Mean_Rating", ascending=False
).head(10)

print("\n===== TOP-RATED LOCATIONS (MINIMUM 20 REVIEWS) =====")
print(top_locations.round(3))

if not top_locations.empty:
    plt.figure(figsize=(11, 6))
    plt.bar(
        top_locations.index,
        top_locations["Mean_Rating"]
    )
    plt.title("10 Highest-Rated Company Locations")
    plt.xlabel("Company Location")
    plt.ylabel("Mean Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No locations have at least 20 valid reviews.")

# 6. Pie chart: Distribution of rating classes
# Low: rating < 3
# Medium: 3 <= rating < 4
# High: rating >= 4
def rating_class(rating):
    if rating < 3:
        return "Low (<3)"
    elif rating < 4:
        return "Medium (3 to <4)"
    else:
        return "High (>=4)"

rating_classes = rating_data["Rating"].apply(rating_class)
class_counts = rating_classes.value_counts().reindex(
    ["Low (<3)", "Medium (3 to <4)", "High (>=4)"],
    fill_value=0
)

plt.figure(figsize=(7, 7))
plt.pie(
    class_counts,
    labels=class_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Distribution of Cocoa Rating Classes")
plt.tight_layout()
plt.show()

print("\nRating class counts:")
print(class_counts)

# 7. Histogram: Cocoa percentage distribution
cocoa_data = df["Cocoa Percent"].dropna()

plt.figure(figsize=(9, 5))
plt.hist(cocoa_data, bins=10, edgecolor="black")
plt.title("Distribution of Cocoa Percentages")
plt.xlabel("Cocoa Percent")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 8. Scatter plot: Review year versus mean rating per year
mean_rating_per_year = rating_data.groupby("Review Date")["Rating"].mean()

plt.figure(figsize=(9, 5))
plt.scatter(
    mean_rating_per_year.index,
    mean_rating_per_year.values
)
plt.title("Review Year vs Mean Cocoa Rating")
plt.xlabel("Review Year")
plt.ylabel("Mean Rating")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nMean rating per year:")
print(mean_rating_per_year.round(3))


# Explanation:
#
# 1. Dataset characteristics:
#    shape gives the number of rows and columns.
#    dtypes displays the data type of each column.
#    head displays the first five records.
#
# 2. Line chart:
#    The x-axis represents review year and the y-axis represents
#    the number of reviews in that year.
#
# 3. Bar chart:
#    The mean rating is calculated for each Company Location.
#    Locations with fewer than 20 valid reviews are excluded.
#    The 10 locations with the highest mean ratings are displayed.
#
# 4. Pie chart:
#    Ratings are divided into three classes:
#    Low (<3), Medium (3 to less than 4), and High (>=4).
#    The pie chart shows the percentage of reviews in each class.
#
# 5. Histogram:
#    Shows the frequency distribution of Cocoa Percent.
#    Each bar represents the number of reviews within a percentage range.
#
# 6. Scatter plot:
#    Each point represents a review year and its mean rating.
#    This helps visualise changes in average rating over time.
#
# 7. Attribute types:
#    Company and Company Location are categorical attributes.
#    Rating and Cocoa Percent are numerical attributes.
#    Review Date represents the review year and is treated numerically here.
#
# Conclusion:
# These visualizations help understand review trends, location ratings,
# rating distribution, cocoa percentage distribution, and yearly changes
# in average ratings.
