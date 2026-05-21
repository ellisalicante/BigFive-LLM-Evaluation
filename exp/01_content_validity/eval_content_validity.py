
### 0) IMPORTS
from src.content_validity_metrics import *
pd.set_option('display.max_columns', None)



### 1) READ DATA
# Ratings from 3 experts
ratings = pd.read_csv("../../dat/01_content_validity/expert_ratings.csv")



### 2) CONTENT VALIDITY

# Experts rated content validity wrt. three dimensions:
facets = ["suitability_facet", "suitability_chatbots", "clarity"]

for facet in facets:
    print(f"\n\nFacet: {facet}")
    ratings_group = ratings.groupby("questionnaire")[facet]

    # Calculate content validity metrics
    mean_df = ratings_group.apply(mean)
    sd_df = ratings_group.apply(sd)

    gwet_list, sd_list, icvi_list, scvi_list = [], [], [], []

    for q, df in ratings.groupby("questionnaire"):
        sd_list.append((q, sd(df[facet])))
        gwet_list.append((q,   gwet_ac2(df, facet)))
        icvi_vals = i_cvi(df, facet)
        icvi_list.append((q, round(icvi_vals.mean(), 2)))
        scvi_list.append((q, s_cvi(df, facet)))

    sd_df = list_to_df(sd_list, "SD")
    gwet_df = list_to_df(gwet_list, "Gwet's AC2")
    scvi_df = list_to_df(scvi_list, "S-CVI")

    # Save in summary df and print
    summary = pd.concat([
        mean_df.rename("M"),
        sd_df,
        scvi_df,
        gwet_df
    ], axis=1)

    print(summary)
