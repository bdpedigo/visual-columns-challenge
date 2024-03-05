# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspologic.embed import ClassicalMDS

test = False
n_columns = 796
add_fake_nodes = True
max_iter = 101
class_weight = 175
n_init = 1
tol = 0.001
sparse = True
save_name = "visual-columns-challenge/results/test=False-class_weight=100-tol=0.001-max_iter=101-sparse=True"
score_df = pd.read_csv(f"{save_name}_scores.csv", index_col=0)
# score_df.index.name = "iteration"
score_df = score_df.reset_index()


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=score_df, x="iteration", y="n_within_group")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=score_df, x="iteration", y="swaps_from_last")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=score_df, x="iteration", y="n_matched")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=score_df, x="iteration", y="n_violations")

# %%

with open(f"{save_name}_results_by_iter.pkl", "rb") as f:
    results_by_iter = pickle.load(f)

# %%
all_perms = []
for result in results_by_iter:
    perm = result.indices_B
    all_perms.append(perm)

# %%

all_diff_df = pd.DataFrame(
    index=range(len(all_perms)), columns=range(len(all_perms)), dtype=float
).fillna(0.0)
for i in range(len(all_perms)):
    for j in range(i + 1, len(all_perms)):
        diff = np.linalg.norm(all_perms[i] - all_perms[j], ord=1)
        all_diff_df.loc[i, j] = diff
        all_diff_df.loc[j, i] = diff

# %%

sns.heatmap(all_diff_df, cmap="viridis")

# %%

diff_df = all_diff_df.loc[10:, 10:]

scores = pd.Series(
    index=np.arange(len(results_by_iter)),
    data=[result.score for result in results_by_iter],
    dtype=float,
)

# %%
embedding = ClassicalMDS(n_components=2, dissimilarity="precomputed")
X = embedding.fit_transform(diff_df.values)

path_df = pd.DataFrame(X, columns=["x", "y"], index=diff_df.index)
path_df["iteration"] = path_df.index
path_df["score"] = scores

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=path_df, x="x", y="y", hue="iteration", palette="viridis", ax=ax)

# plot lines connecting all of the iterations in order
for i in path_df.index[:-1]:
    x1, y1 = path_df.loc[i, ["x", "y"]]
    x2, y2 = path_df.loc[i + 1, ["x", "y"]]
    ax.plot([x1, x2], [y1, y2], color="black", alpha=0.2, linewidth=0.5)


sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25, 1), title="Iteration")

ax.set_xlabel("MDS 1")
ax.set_ylabel("MDS 2")
ax.set_xticks([])
ax.set_yticks([])

# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=path_df, x="x", y="y", hue="score", palette="Greys", ax=ax)

# plot lines connecting all of the iterations in order
for i in path_df.index[:-1]:
    x1, y1 = path_df.loc[i, ["x", "y"]]
    x2, y2 = path_df.loc[i + 1, ["x", "y"]]
    ax.plot([x1, x2], [y1, y2], color="black", alpha=0.2, linewidth=0.5)


sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25, 1), title="Score")

ax.set_xlabel("MDS 1")
ax.set_ylabel("MDS 2")
ax.set_xticks([])
ax.set_yticks([])

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.scatterplot(data=path_df, x="iteration", y="y", hue="score", palette="Greys", ax=ax)

for i in path_df.index[:-1]:
    x1, y1 = path_df.loc[i, ["iteration", "y"]]
    x2, y2 = path_df.loc[i + 1, ["iteration", "y"]]
    ax.plot([x1, x2], [y1, y2], color="black", alpha=0.2, linewidth=0.5)

ax.set_xlabel("Iteration")
ax.set_ylabel("MDS 2")
ax.set_xticks([])
ax.set_yticks([])

sns.move_legend(ax, "upper right", bbox_to_anchor=(1.35, 1), title="Score")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.scatterplot(
    data=path_df, x="y", y="score", hue="iteration", palette="viridis", ax=ax
)

# for i in path_df.index[:-1]:
#     x1, y1 = path_df.loc[i, ["iteration", "y"]]
#     x2, y2 = path_df.loc[i + 1, ["iteration", "y"]]
#     ax.plot([x1, x2], [y1, y2], color="black", alpha=0.2, linewidth=0.5)

ax.set_xlabel("Score")
ax.set_xlabel("MDS 2")
# ax.set_xticks([])
# ax.set_yticks([])

sns.move_legend(ax, "upper right", bbox_to_anchor=(1.35, 1), title="Iteration")

# %%
