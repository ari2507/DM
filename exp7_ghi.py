
# Question:
# Implement the Apriori algorithm on the GHI dataset.
# Discretize Economy, Fam, Health, Freedom, and H_Score into
# Low, Medium, and High using equal-frequency binning.
# Treat each country as a transaction, find frequent itemsets
# using minimum support = 0.15, and verify results using mlxtend.
# %pip install mlxtend

import pandas as pd
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as mlxtend_apriori

# Load dataset
df = pd.read_csv("GHI_Report (1).csv")
df.columns = df.columns.str.strip()

columns = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

# Convert attributes to numeric and remove missing values
for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=columns).reset_index(drop=True)
print("Number of countries:", len(df))

# Discretize each attribute into Low, Medium, and High
for col in columns:
    categories = pd.qcut(
        df[col].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"]
    )
    df[col + "_Level"] = col + "-" + categories.astype(str)

item_columns = [col + "_Level" for col in columns]

# Create one transaction per country
transactions = df[item_columns].values.tolist()

print("\nFirst 5 transactions:")
for transaction in transactions[:5]:
    print(transaction)

# Set minimum support
min_support = 0.15
n = len(transactions)

# Calculate support of an itemset
def calculate_support(itemset):
    count = sum(
        1 for transaction in transactions
        if itemset.issubset(set(transaction))
    )
    return count / n

# Find frequent 1-itemsets
items = sorted(set(item for t in transactions for item in t))
L1 = {}

for item in items:
    itemset = frozenset([item])
    s = calculate_support(itemset)
    if s >= min_support:
        L1[itemset] = s

frequent_levels = {1: L1}

# Generate candidate itemsets using the Apriori principle
def generate_candidates(previous_level, k):
    candidates = set()
    previous_itemsets = list(previous_level.keys())

    for i in range(len(previous_itemsets)):
        for j in range(i + 1, len(previous_itemsets)):
            candidate = previous_itemsets[i] | previous_itemsets[j]

            if len(candidate) != k:
                continue

            # Every (k-1)-subset must be frequent
            if all(
                frozenset(subset) in previous_level
                for subset in combinations(sorted(candidate), k - 1)
            ):
                candidates.add(candidate)

    return candidates

# Find larger frequent itemsets
k = 2

while frequent_levels[k - 1]:
    candidates = generate_candidates(frequent_levels[k - 1], k)
    current_level = {}

    for candidate in candidates:
        s = calculate_support(candidate)
        if s >= min_support:
            current_level[candidate] = s

    if not current_level:
        break

    frequent_levels[k] = current_level
    k += 1

# Display frequent itemsets found by our implementation
print("\n===== APRIORI FROM SCRATCH =====")
print("Minimum support:", min_support)

for level in sorted(frequent_levels):
    print(f"\nL{level}: Frequent {level}-itemsets")

    for itemset, s in sorted(
        frequent_levels[level].items(),
        key=lambda x: sorted(x[0])
    ):
        print(sorted(itemset), "Support:", round(s, 4))

# Verify results using mlxtend
encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)

transaction_df = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

library_result = mlxtend_apriori(
    transaction_df,
    min_support=min_support,
    use_colnames=True
)

library_itemsets = {
    frozenset(row["itemsets"]): row["support"]
    for _, row in library_result.iterrows()
}

our_itemsets = {}
for level in frequent_levels.values():
    our_itemsets.update(level)

print("\n===== VERIFICATION WITH MLXTEND =====")
print("Our frequent itemsets:", len(our_itemsets))
print("mlxtend frequent itemsets:", len(library_itemsets))

if our_itemsets.keys() == library_itemsets.keys():
    print("PASS: Both implementations found the same itemsets.")

    support_matches = all(
        abs(our_itemsets[itemset] - library_itemsets[itemset]) < 1e-9
        for itemset in our_itemsets
    )

    if support_matches:
        print("PASS: All support values match.")
    else:
        print("WARNING: Some support values differ.")
else:
    print("WARNING: The frequent itemsets do not match.")

# Directly verify support for up to five frequent itemsets
print("\n===== DIRECT SUPPORT CHECK =====")
checked = 0

for itemset, s in sorted(
    our_itemsets.items(),
    key=lambda x: (len(x[0]), sorted(x[0]))
):
    direct_support = calculate_support(itemset)

    print(
        sorted(itemset),
        "Calculated support:", round(direct_support, 4),
        "Stored support:", round(s, 4)
    )

    checked += 1
    if checked == 5:
        break


# Explanation:
# 1. Apriori finds frequent combinations of items in transactions.
# 2. The five numerical GHI attributes are divided into Low, Medium,
#    and High categories using equal-frequency binning.
# 3. Each country becomes one transaction containing five categorical items.
# 4. Support = Number of transactions containing an itemset / Total transactions.
# 5. Minimum support is 0.15, so an itemset must appear in at least 15%
#    of the countries to be considered frequent.
# 6. L1 contains frequent one-itemsets, L2 contains frequent pairs,
#    and L3 and subsequent levels contain larger frequent itemsets.
# 7. The Apriori principle states that every non-empty subset of a
#    frequent itemset must also be frequent. This allows candidates to be pruned.
# 8. mlxtend is used to independently verify the itemsets and support values.
# 9. The direct support check recalculates support from the original transactions.
# 10. Conclusion: The program identifies frequent GHI attribute combinations
#     and checks the custom Apriori implementation against mlxtend.
