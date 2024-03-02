# %%
import glob
import os

import pandas as pd

result_dir = "visual-columns-challenge/results"

# use glob to find all files that end in _scores.csv
score_files = glob.glob(os.path.join(result_dir, "*_scores.csv"))

# now create a dataframe to keep track of the files
file_df = pd.DataFrame(score_files, columns=["file_path"])
file_df["file_name"] = file_df["file_path"].apply(lambda x: os.path.basename(x))

# find all the instances of 'test=' and then get the value after it, as a column
file_df["test"] = file_df["file_name"].apply(
    lambda x: x.split("test=")[1].split("-")[0]
)

# find all the instances of 'class_weight=' and then get the value after it, as a column
file_df["class_weight"] = file_df["file_name"].apply(
    lambda x: x.split("class_weight=")[1].split("-")[0]
)

# find all the instances of 'max_iter=' and then get the value after it, as a column
file_df["max_iter"] = file_df["file_name"].apply(
    lambda x: x.split("max_iter=")[1].split("-")[0]
)

real_file_df = file_df[file_df["test"] == "False"]

# %%

dfs = []
keys = ["class_weight"]
# now we can loop through the real_file_df and load the scores
for i, row in real_file_df.iterrows():
    score_df = pd.read_csv(row["file_path"], index_col=0)
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
all_scores = all_scores.reset_index(drop=True)
# %%
baseline_n_synapses = 1_383_531
baseline_n_matched = 22_578
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_kws = dict(color="black", linestyle="--", label="Benchmark", linewidth=1.5)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=all_scores, x="iteration", y="n_within_group", hue="class_weight")

ax.set_ylabel("Within group synapses")
ax.set_xlabel("Iteration")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(
    data=all_scores, x="iteration", y="corrected_n_within_group", hue="class_weight"
)

ax.set_ylabel("Within group synapses")
ax.set_xlabel("Iteration")

ax.axhline(baseline_n_synapses, **benchmark_kws)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=all_scores, x="iteration", y="n_matched", hue="class_weight")

ax.set_ylabel("# cells matched")
ax.set_xlabel("Iteration")

ax.axhline(baseline_n_matched, **benchmark_kws)

# %%
