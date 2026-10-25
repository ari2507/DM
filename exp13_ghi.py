# Part B - K-Means (Exp 13), GHI: Run K-Means on the four factors (Z-score normalised) for k = 2 to 6. Plot the elbow curve (SSE vs k) and the silhouette score vs k, choose the best k, and plot Economy vs Health coloured by the final clusters.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("GHI_Report (1).csv")

# Select four factors
cols = ["Economy", "Fam", "Health", "Freedom"]
X = StandardScaler().fit_transform(df[cols])

# Calculate SSE and silhouette scores
sse = []
silhouette = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sse.append(model.inertia_)
    silhouette.append(silhouette_score(X, labels))

# Plot elbow curve
plt.plot(range(2, 7), sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()

# Plot silhouette scores
plt.plot(range(2, 7), silhouette, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()

# Choose best k using highest silhouette score
best_k = range(2, 7)[silhouette.index(max(silhouette))]
print("Best k =", best_k)

# Final K-Means model
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

# Plot Economy vs Health
plt.scatter(df["Economy"], df["Health"],
            c=df["Cluster"], cmap="viridis")
plt.xlabel("Economy")
plt.ylabel("Health")
plt.title("Economy vs Health - K-Means Clusters")
plt.colorbar(label="Cluster")
plt.show()

# . Aim of the experiment
# To apply K-Means clustering to four standardised factors in the GHI dataset, evaluate cluster counts from \(k=2\) to \(k=6\) using the elbow method and silhouette score, select a suitable number of clusters, and visualise the final clusters using Economy versus Health.
# 2. What is clustering?
# Clustering is an unsupervised machine learning technique that groups similar data points together.

# Unsupervised means that the algorithm is not given predefined class labels. It discovers groups from the numerical features.

# For example, using economic and health indicators, K-Means may group observations with similar development characteristics.
# - Observations in the same cluster should be relatively similar.
# - Observations in different clusters should be relatively different.
# 3. What is K-Means?
# K-Means is a clustering algorithm that divides a dataset into \(k\) clusters.

# The letter \(k\) represents the number of clusters chosen by the user.

# For example:
# - \(k=2\): two clusters.
# - \(k=3\): three clusters.
# - \(k=4\): four clusters.
# K-Means uses the centroid, or mean position, of each cluster to represent that cluster.
# Steps of the K-Means algorithm

# Step 1: Select the features

# Choose the four numerical GHI factors: Economy, Family, Health, and Freedom.




# Step 2: Standardise the features

# Convert the four factors to comparable scales using Z-score normalisation.




# Step 3: Initialise centroids

# Select \(k\) initial cluster centres.




# Step 4: Assign data points

# Assign each observation to its nearest centroid using Euclidean distance.




# Step 5: Update centroids

# Calculate the mean of the observations in each cluster and move the centroid to that mean.




# Step 6: Repeat

# Repeat assignment and centroid updating until the solution stabilises or the iteration limit is reached.



# 4. Why do we apply Z-score normalisation before K-Means?
# K-Means relies on Euclidean distance. If the features have very different scales, a feature with a larger numerical range can dominate the distance calculation.

# For example, suppose Economy ranges from 0 to 2, while another feature ranges from 0 to 100. Without scaling, the second feature may have a much greater influence on the clustering.

# Standardisation gives each feature a mean of approximately 0 and a standard deviation of approximately 1.
# Formula
# \[
# z=\frac{x-\mu}{\sigma}
# \]

# Code
# from sklearn.preprocessing import StandardScalercols = ["Economy", "Fam", "Health", "Freedom"]X = StandardScaler().fit_transform(df[cols])



# Explanation:
# - cols specifies the four features used for clustering.
# - df[cols] extracts those columns.
# - StandardScaler() creates the scaler.
# - fit_transform() calculates the mean and standard deviation and standardises the selected data.
# - X stores the scaled numerical array used by K-Means.
# Note: The column names must exactly match those in your CSV. If your file uses names such as Economy (GDP per Capita), Family, or Health (Life Expectancy), change cols to match them.
# 5. Euclidean distance formula
# K-Means assigns each observation to the nearest centroid using Euclidean distance.

# For a point with four features and a centroid with four corresponding features:
# \[
# d=\sqrt{\sum_{j=1}^{4}(x_j-c_j)^2}
# \]

# Expanded:
# \[
# d=\sqrt{
# (x_1-c_1)^2+
# (x_2-c_2)^2+
# (x_3-c_3)^2+
# (x_4-c_4)^2}
# \]

# Where:
# - \(x_j\) = the observation's value for feature \(j\).
# - \(c_j\) = the centroid's value for feature \(j\).
# - \(d\) = distance between the observation and centroid.
# The observation is assigned to the centroid with the smallest distance.
# 6. Elbow method and SSE
# The elbow method helps identify a suitable number of clusters by comparing the number of clusters with the clustering error.

# SSE stands for Sum of Squared Errors. In K-Means, it is the sum of squared distances between observations and their assigned cluster centroids. It is also called within-cluster sum of squares or inertia in scikit-learn.
# Formula
# \[
# SSE=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2
# \]

# Where:
# - \(n\) = number of observations.
# - \(x_i\) = the feature vector for observation \(i\).
# - \(\mu_{c_i}\) = the centroid of the cluster assigned to observation \(i\).
# - \(\|\cdot\|^2\) = squared Euclidean distance.
# Example
# Suppose three observations are assigned to a cluster with centroid 4:
# Observation	Distance from centroid	Squared distance
# 3	1	1
# 4	0	0
# 6	2	4
# Total SSE		5

# The total SSE for a full clustering is calculated by adding the squared distances for every observation across all clusters.
# Why does SSE decrease as \(k\) increases?
# When more clusters are used, observations can generally be assigned to closer centroids. Therefore, SSE decreases or stays the same as \(k\) increases.

# However, choosing the largest \(k\) is not necessarily useful. Too many clusters can make the grouping unnecessarily complicated.
# What is the elbow point?
# The elbow is the point where the SSE curve changes from decreasing steeply to decreasing more gradually.







# kdnuggets.com


# The bend indicates that adding more clusters gives diminishing improvements. The elbow is a visual guideline, not always a clearly defined point.
# Code
# sse = []for k in range(2, 7):    model = KMeans(n_clusters=k, random_state=42, n_init=10)    labels = model.fit_predict(X)    sse.append(model.inertia_)plt.plot(range(2, 7), sse, marker="o")plt.xlabel("Number of clusters (k)")plt.ylabel("SSE")plt.title("Elbow Method")plt.show()



# Line-by-line explanation:
# 1. sse = [] creates an empty list to store the SSE values.
# 2. range(2, 7) tests \(k=2,3,4,5,6\).
# 3. KMeans(n_clusters=k, ...) creates a K-Means model with the current number of clusters.
# 4. random_state=42 makes the initial randomisation reproducible.
# 5. n_init=10 runs the algorithm with 10 different centroid initialisations and keeps the best result by inertia.
# 6. fit_predict(X) fits the model and returns the cluster label for every observation.
# 7. model.inertia_ gives the final SSE.
# 8. plt.plot() plots SSE against \(k\).
# 9. The axis labels and title explain the graph.
# 7. Silhouette score
# The silhouette score measures how well an observation fits into its assigned cluster compared with the nearest alternative cluster.

# It considers two things:
# - Cohesion: How close an observation is to other observations in its own cluster.
# - Separation: How far it is from observations in the nearest different cluster.
# Formula
# For an observation \(i\):
# \[
# s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
# \]

# Where:
# - \(a(i)\) = average distance from observation \(i\) to other observations in its own cluster.
# - \(b(i)\) = lowest average distance from observation \(i\) to observations in another cluster, considering all other clusters.
# - \(s(i)\) = silhouette score for observation \(i\).
# The overall silhouette score is the mean of the individual observations' silhouette scores.
# Range and interpretation
# Score	Interpretation
# Close to +1	Clusters are well separated
# Close to 0	Clusters overlap or observations lie near boundaries
# Close to -1	Some observations may be assigned to the wrong cluster

# A higher overall score generally indicates better-separated, more cohesive clusters.







# blog.dailydoseofds.com


# Code
# from sklearn.metrics import silhouette_scoresilhouette = []for k in range(2, 7):    model = KMeans(n_clusters=k, random_state=42, n_init=10)    labels = model.fit_predict(X)    silhouette.append(silhouette_score(X, labels))plt.plot(range(2, 7), silhouette, marker="o")plt.xlabel("Number of clusters (k)")plt.ylabel("Silhouette Score")plt.title("Silhouette Score vs K")plt.show()



# Explanation:
# 1. silhouette = [] stores the scores.
# 2. The loop tests cluster counts from 2 to 6.
# 3. fit_predict(X) calculates the clusters.
# 4. silhouette_score(X, labels) calculates the mean silhouette score using the scaled features and predicted cluster labels.
# 5. plt.plot() displays how the score changes with \(k\).
# Why start at \(k=2\)? The silhouette score requires at least two clusters and fewer clusters than observations. It cannot evaluate a single-cluster solution.
# 8. How do we select the best value of \(k\)?
# The experiment tests five possibilities: \(k=2,3,4,5,6\).

# Use both graphs to make the decision:
# 1. Find the elbow in the SSE curve.
# 2. Find the highest silhouette score.
# 3. Compare the two results and select a reasonable \(k\) that gives compact, well-separated clusters without unnecessary complexity.
# The highest silhouette score is a useful starting point, but it does not always guarantee the most meaningful clusters for your application.
# Code
# best_k = range(2, 7)[silhouette.index(max(silhouette))]print("Best k =", best_k)



# Explanation:
# - max(silhouette) finds the highest score.
# - silhouette.index(...) finds the position of that score in the list.
# - range(2, 7)[...] maps the position back to the corresponding \(k\).
# - print() displays the selected value.
# For example, if the highest score is at \(k=3\), the output will be:
# Best k = 3


# This is only an example; your actual best \(k\) depends on the dataset.
# 9. Final K-Means clustering
# After selecting \(k\), run the model again using that value to generate the final clusters.
# Code
# model = KMeans(n_clusters=best_k, random_state=42, n_init=10)df["Cluster"] = model.fit_predict(X)



# Explanation:
# - n_clusters=best_k uses the selected number of clusters.
# - fit_predict(X) trains the model and assigns each observation a cluster label.
# - df["Cluster"] adds those labels to the original DataFrame.
# For example, a cluster label may be 0, 1, or 2 if three clusters were selected.

# These labels are not rankings. Cluster 2 is not necessarily better than cluster 1; the numbers simply identify different groups.
# 10. Plot Economy versus Health
# The final graph shows the relationship between the Economy and Health factors, with each point coloured according to its cluster.
# Code
# plt.scatter(df["Economy"], df["Health"],            c=df["Cluster"], cmap="viridis")plt.xlabel("Economy")plt.ylabel("Health")plt.title("Economy vs Health - K-Means Clusters")plt.colorbar(label="Cluster")plt.show()



# Explanation:
# - plt.scatter() creates a scatter plot.
# - df["Economy"] provides the horizontal-axis values.
# - df["Health"] provides the vertical-axis values.
# - c=df["Cluster"] assigns colours based on cluster labels.
# - cmap="viridis" specifies the colour map.
# - plt.xlabel() and plt.ylabel() label the axes.
# - plt.title() adds a title.
# - plt.colorbar() explains the mapping between colours and cluster labels.
# - plt.show() displays the final graph.







# datapilots.tistory.com


# How to interpret the scatter plot
# - Points close together generally have similar Economy and Health values.
# - Different colours indicate different cluster assignments.
# - Clearly separated colour groups suggest that the two displayed features distinguish some of the clusters.
# - Overlapping colours can occur because K-Means used four features, not just Economy and Health.
# The graph is a two-dimensional view of a clustering model built in four-dimensional feature space. Two clusters may look overlapping in this graph while being distinct when Family and Freedom are also considered.

# 11. Important formulas to remember
# Concept	Formula
# Min-Max normalisation	\(\displaystyle x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}\)
# Z-score normalisation	\(\displaystyle z=\frac{x-\mu}{\sigma}\)
# Euclidean distance	\(\displaystyle d=\sqrt{\sum_{j=1}^{p}(x_j-c_j)^2}\)
# K-Means SSE	\(\displaystyle SSE=\sum_{i=1}^{n}\|x_i-\mu_{c_i}\|^2\)
# Silhouette score	\(\displaystyle s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}\)

# Here, \(p\) is the number of features, \(n\) is the number of observations, and \(c_i\) denotes the cluster assigned to observation \(i\).
# 12. Common viva questions and answers
# Q1. What is the difference between normalisation and standardisation?

# Normalisation often rescales values to a fixed range such as 0–1. Standardisation transforms values to have a mean of approximately 0 and a standard deviation of approximately 1.

# Q2. Does normalisation change the distribution shape?

# It does not change the underlying distribution shape under these positive linear transformations, though histogram appearance can vary with binning and axes.

# Q3. Why is feature scaling important in K-Means?

# K-Means uses distances. Scaling prevents features with larger numerical ranges from dominating distance calculations.

# Q4. What does \(k\) mean in K-Means?

# It is the number of clusters into which the algorithm divides the observations.

# Q5. What is a centroid?

# A centroid is the mean position of the observations assigned to a cluster.

# Q6. What is SSE?

# SSE is the sum of squared distances between observations and their assigned centroids. Lower SSE indicates more compact clusters, but it normally decreases as \(k\) increases.

# Q7. What is the elbow method?

# It plots SSE against \(k\) and looks for the point where the reduction in SSE starts slowing down.

# Q8. What is the silhouette score?

# It measures how well observations fit their own clusters compared with the nearest alternative cluster. Values range from -1 to +1.

# Q9. Why do we test \(k=2\) to \(6\)?

# To compare several possible cluster counts and identify a reasonable solution using the elbow curve and silhouette scores.

# Q10. Is K-Means supervised or unsupervised?

# It is unsupervised because it groups observations without requiring predefined target labels.