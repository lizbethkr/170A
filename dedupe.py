import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

cols = ['id','name','first_name','middle_name','last_name','affiliation',
        'homepage','scholarid','clean_author_name']

df = pd.read_csv(
    "data/processed/transformed_csrankings.csv",
    na_values=["\\N"],
    names=cols,
    skipinitialspace=True,
    encoding="utf-8-sig"
)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Fill missing middle names with empty string
df['middle_name'] = df['middle_name'].fillna("")

print("Columns loaded:", df.columns.tolist())
print(df.head())

# Splink Setup
db_api = DuckDBAPI()

settings = SettingsCreator(
    link_type="dedupe_only",
    unique_id_column_name="id",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.JaroAtThresholds("last_name"),
        cl.ExactMatch("affiliation").configure(term_frequency_adjustments=True),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "last_name"),
        block_on("last_name")
    ]
)

linker = Linker(df, settings, db_api)

# Training 
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "last_name")],
    recall=0.7
)

linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)

linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "last_name")
)

# Predict pairwise matches
pairwise_predictions = linker.inference.predict(threshold_match_weight=-5)

# Cluster pairwise predictions
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions, 0.95
)

# Top 10 predicted duplicate pairs
#df_clusters = clusters.as_pandas_dataframe(limit=10)
#print("\nTop 10 predicted duplicate author clusters:")
#print(df_clusters.head(10))

# Group by cluster_id and collect all author names in a list
df_cleaned = clusters.as_pandas_dataframe(limit=None)
cluster_groups = df_cleaned.groupby("cluster_id")["clean_author_name"].apply(list).reset_index()

# Keep only clusters with more than one name (actual duplicates)
duplicate_clusters = cluster_groups[cluster_groups["clean_author_name"].apply(len) > 1]

# Show top 10 clusters with duplicates
print(duplicate_clusters.head(10))

# Cleaned DataFrame with cluster IDs
df_cleaned = df_cleaned.sort_values("id").groupby("cluster_id").first().reset_index()
df_cleaned.to_csv("data/processed/csranking_cleaned_authors.csv", index=False)
print("\nCleaned csranking table saved as 'data/processed/csranking_cleaned_authors.csv'.")