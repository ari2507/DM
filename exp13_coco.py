
# Question:
# Apply K-Means clustering to the Cocoa dataset.
# Group companies having at least 5 reviews using:
# 1. Mean Rating
# 2. Mean Cocoa Percent
# 3. Number of Reviews
# Normalize the features, perform K-Means with k=3,
# plot the clusters, calculate the silhouette coefficient,
# and describe each cluster.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the Cocoa dataset
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")
df.columns = df.columns.str.strip()

# 2. Clean the required columns
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df["Cocoa Percent"] = (
    df["Cocoa Percent"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["Cocoa Percent"] = pd.to_numeric(
    df["Cocoa Percent"], errors="coerce"
)

# Remove rows with missing required values
df = df.dropna(
    subset=["Company", "Rating", "Cocoa Percent"]
).copy()

# Remove empty company names
df = df[df["Company"].astype(str).str.strip() != ""]

# 3. Aggregate data company-wise
company_data = df.groupby("Company").agg(
    Mean_Rating=("Rating", "mean"),
    Mean_Cocoa_Percent=("Cocoa Percent", "mean"),
    Number_of_Reviews=("Rating", "count")
).reset_index()

# 4. Keep companies with at least 5 reviews
company_data = company_data[
    company_data["Number_of_Reviews"] >= 5
].reset_index(drop=True)

print("Number of companies with at least 5 reviews:",
      len(company_data))

if len(company_data) < 3:
    raise ValueError(
        "At least 3 companies with 5 or more reviews are required."
    )

# 5. Select features for clustering
features = [
    "Mean_Rating",
    "Mean_Cocoa_Percent",
    "Number_of_Reviews"
]

X = company_data[features]

# 6. Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Apply K-Means clustering with k=3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

company_data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nFirst 10 companies with cluster labels:")
print(company_data.head(10).round(2).to_string(index=False))

# 8. Calculate silhouette coefficient
labels = company_data["Cluster"]

if 1 < labels.nunique() < len(company_data):
    score = silhouette_score(X_scaled, labels)
    print("\nSilhouette Coefficient:", round(score, 4))
else:
    print("\nSilhouette coefficient cannot be calculated.")

# 9. Display cluster descriptions using cluster averages
cluster_profile = company_data.groupby("Cluster").agg(
    Number_of_Companies=("Company", "count"),
    Mean_Rating=("Mean_Rating", "mean"),
    Mean_Cocoa_Percent=("Mean_Cocoa_Percent", "mean"),
    Average_Reviews=("Number_of_Reviews", "mean")
).round(2)

print("\nCluster Profiles:")
print(cluster_profile.to_string())

# 10. Describe each cluster automatically
print("\nCluster Interpretations:")

overall_rating = company_data["Mean_Rating"].mean()
overall_cocoa = company_data["Mean_Cocoa_Percent"].mean()
overall_reviews = company_data["Number_of_Reviews"].mean()

for cluster, row in cluster_profile.iterrows():
    rating_desc = (
        "higher-than-average ratings"
        if row["Mean_Rating"] >= overall_rating
        else "lower-than-average ratings"
    )

    cocoa_desc = (
        "higher-than-average cocoa percentage"
        if row["Mean_Cocoa_Percent"] >= overall_cocoa
        else "lower-than-average cocoa percentage"
    )

    reviews_desc = (
        "more reviews than average"
        if row["Average_Reviews"] >= overall_reviews
        else "fewer reviews than average"
    )

    print(
        f"Cluster {cluster}: {rating_desc}, "
        f"{cocoa_desc}, and {reviews_desc}."
    )

# 11. Plot clusters: Mean Cocoa Percent vs Mean Rating
plt.figure(figsize=(9, 6))

for cluster in sorted(company_data["Cluster"].unique()):
    subset = company_data[company_data["Cluster"] == cluster]

    plt.scatter(
        subset["Mean_Cocoa_Percent"],
        subset["Mean_Rating"],
        label=f"Cluster {cluster}",
        s=60
    )

plt.xlabel("Mean Cocoa Percent")
plt.ylabel("Mean Rating")
plt.title("K-Means Clustering of Cocoa Companies (k=3)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Explanation:
#
# 1. Data preprocessing:
#    Load the Cocoa CSV, convert Rating and Cocoa Percent to numbers,
#    and remove records with missing required values.
#
# 2. Company-wise aggregation:
#    Mean_Rating = average rating for each company.
#    Mean_Cocoa_Percent = average cocoa percentage for each company.
#    Number_of_Reviews = number of valid ratings for each company.
#
# 3. Filtering:
#    Only companies with at least 5 valid reviews are retained.
#
# 4. Normalization:
#    StandardScaler transforms each feature using:
#    Z = (X - Mean) / Standard Deviation
#    This prevents features with larger numerical scales from
#    dominating the clustering process.
#
# 5. K-Means:
#    K-Means divides companies into 3 clusters based on their
#    normalized features. The algorithm assigns each company to
#    the nearest cluster centre and updates the centres iteratively.
#
# 6. Silhouette coefficient:
#    The score ranges from -1 to 1.
#    Near 1: clusters are well separated.
#    Near 0: clusters overlap.
#    Below 0: some companies may be assigned to an unsuitable cluster.
#
# 7. Cluster profiles:
#    The average rating, cocoa percentage, and review count are
#    compared across clusters to describe their characteristics.
#
# 8. Scatter plot:
#    The x-axis shows mean cocoa percentage and the y-axis shows
#    mean rating. Each colour represents a cluster.
#    Review count is used in clustering but is not shown on these axes.
#
# 9. Conclusion:
#    K-Means groups companies with similar rating, cocoa percentage,
#    and review-count characteristics. The cluster profiles and
#    silhouette coefficient help evaluate and interpret the results.
