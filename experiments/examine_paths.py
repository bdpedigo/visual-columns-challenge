# %%
from pathlib import Path

import pandas as pd

result_dir = Path("visual-columns-challenge/results/")

experiment_manifest = pd.read_csv(result_dir / "experiment_manifest.csv").set_index(
    "experiment_id"
)
# %%
# experiment_ids = [
#     1712005849,
#     1712096956,
#     1712153817,
# ]
experiment_ids = experiment_manifest.index

all_scores = []
for experiment_id in experiment_ids:
    try:
        experiment_scores = pd.read_csv(
            result_dir / str(experiment_id) / f"{experiment_id}_scores.csv", index_col=0
        )
        all_scores.append(experiment_scores)
    except FileNotFoundError:
        print(experiment_id)

all_scores = pd.concat(all_scores, ignore_index=True)

# %%
import seaborn as sns

sns.lineplot(
    data=all_scores.query("iteration > 5"),
    x="iteration",
    y="corrected_n_within_group",
    hue="class_weight",
)

#%%
all_scores['corrected_n_within_group'].max()

#%%
idx = all_scores['corrected_n_within_group'].idxmax()

all_scores.loc[idx]

# %%
all_scores.query("iteration > 5").groupby("class_weight")[
    "corrected_n_within_group"
].idxmax()
