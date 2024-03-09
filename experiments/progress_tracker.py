# %%
import glob
import os
import pickle
from pathlib import Path

import pandas as pd

result_dir = Path("visual-columns-challenge/results")

# use glob to find all files that end in _scores.csv
score_files = glob.glob(os.path.join(result_dir, "*_scores.csv"))

# now create a dataframe to keep track of the files
file_df = pd.DataFrame(score_files, columns=["file_path"])
file_df["file_name"] = file_df["file_path"].apply(lambda x: os.path.basename(x))

# find all the instances of 'test=' and then get the value after it, as a column


def get_var_from_name(x, var):
    if f"{var}=" in x:
        value = x.split(f"{var}=")[1].split("-")[0]
        if value == "True":
            return True
        elif value == "False":
            return False
        else:
            return value
    else:
        return None


file_df["test"] = file_df["file_name"].apply(lambda x: get_var_from_name(x, "test"))


# find all the instances of 'class_weight=' and then get the value after it, as a column
file_df["class_weight"] = file_df["file_name"].apply(
    lambda x: get_var_from_name(x, "class_weight")
)

# find all the instances of 'max_iter=' and then get the value after it, as a column
file_df["max_iter"] = file_df["file_name"].apply(
    lambda x: get_var_from_name(x, "max_iter")
)

file_df["restart"] = file_df["file_name"].apply(
    lambda x: get_var_from_name(x, "restart")
)

real_file_df = file_df.query("test.isna() | test == False")


# %%

dfs = []
keys = ["class_weight", "restart"]
# now we can loop through the real_file_df and load the scores
for i, row in real_file_df.iterrows():
    score_df = pd.read_csv(row["file_path"], index_col=0)
    score_df["file"] = row["file_name"]
    # if "iteration" not in score_df.columns:
    #     score_df["iteration"] = score_df.index + 1
    #     score_df.index.name = "iteration"
    # score_df.index.name = "iteration"
    score_df = score_df.reset_index()
    for key in keys:
        score_df[key] = row[key]
    dfs.append(score_df)

all_scores = pd.concat(dfs, ignore_index=False)
all_scores["iteration"] = all_scores.index.copy() + 1
# all_scores['class_weight'] = all_scores['class_weight'].astype(int)
all_scores = all_scores.reset_index(drop=True)
all_scores = all_scores.query("class_weight != '120'")
# all_scores = all_scores.query("restart.isna() | restart == False")
all_scores["restart"] = all_scores["restart"].fillna(False)


# %%

baseline_n_synapses = 1_383_531
baseline_n_matched = 22_578
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_kws = dict(color="black", linestyle="--", label="Benchmark", linewidth=1.5)

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# sns.lineplot(data=all_scores, x="iteration", y="n_within_group", hue="class_weight")

# ax.set_ylabel("Within group synapses")
# ax.set_xlabel("Iteration")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(
    data=all_scores,
    x="iteration",
    y="corrected_n_within_group",
    hue="class_weight",
    style="restart",
    # estimator=None,
)

ax.set_ylabel("Within group synapses")
ax.set_xlabel("Iteration")

ax.axhline(baseline_n_synapses, **benchmark_kws)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(
    data=all_scores,
    x="iteration",
    y="corrected_n_matched",
    hue="class_weight",
    style="restart",
)

ax.set_ylabel("# cells matched")
ax.set_xlabel("Iteration")

ax.axhline(baseline_n_matched, **benchmark_kws)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
ax = axs[0]
sns.lineplot(
    data=all_scores.query("iteration > 10"),
    x="iteration",
    y="corrected_n_within_group",
    hue="class_weight",
    style="restart",
    ax=ax,
)
ax.get_legend().remove()
ax.set_ylabel("Within group synapses")
ax.set_xlabel("Iteration")

ax = axs[1]
sns.lineplot(
    data=all_scores.query("iteration > 10"),
    x="iteration",
    y="corrected_n_matched",
    hue="class_weight",
    style="restart",
    ax=ax,
)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.5, 1))

ax.set_ylabel("# cells matched")
ax.set_xlabel("Iteration")

# %%
all_scores.sort_values(
    ["corrected_n_within_group", "corrected_n_matched"], ascending=False
)

# %%
loc_max = all_scores["corrected_n_within_group"].idxmax()

row = all_scores.loc[loc_max]
iteration = row["index"]


file_name = row["file"].split("_")[:-1]
file_prefix = "_".join(file_name)

with open(result_dir / f"{file_prefix}_results_by_iter.pkl", "rb") as f:
    results_by_iter = pickle.load(f)

result = results_by_iter[iteration]

# %%
