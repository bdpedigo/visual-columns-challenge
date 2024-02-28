# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from sklearn.metrics import adjusted_rand_score

from graspologic.match import graph_match

DATA_PATH = Path("visual-columns-challenge/data/")

columns_df = pd.read_csv(DATA_PATH / "ol_columns.csv")
columns_df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
columns_df.set_index("cell_id", inplace=True)


def remapper(x):
    if x == "not assigned":
        return None
    return x


columns_df["column_id"] = columns_df["column_id"].map(remapper)
columns_df["column_id"] = columns_df["column_id"].astype("Int64")

connections_df = pd.read_csv(DATA_PATH / "ol_connections.csv")
connections_df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
connections_df.rename(
    columns={"from_cell_id": "source", "to_cell_id": "target", "synapses": "weight"},
    inplace=True,
)

metadata_df = pd.read_csv(DATA_PATH / "ol_metadata.csv")

nf = NetworkFrame(
    nodes=columns_df,
    edges=connections_df,
)

n_columns = 796
test = True
if test:
    n_select_columns = 30
    select_cols = np.random.choice(
        np.arange(1, n_columns + 1), size=n_select_columns, replace=False
    )
    # just for testing
    nf = nf.query_nodes("column_id.isin(@select_cols)", local_dict=locals()).copy()
    n_columns = n_select_columns

n_types = 31
n_ideal_cells = n_columns * n_types

# %%

# TODO redo this to consider how cells types with more than 31 members are dealt with
# possibly means I need to pad B as well...

# TODO look into sweeping over the weighting of AB vs S terms

# TODO FAQ-2-opt: do some kind of optimize, perturb, optimize, perturb scheme
# should be easy to implement with an outer loop and a literal permutation of the input
# myself

# TODO look into an adopted padding version of this? could that matter here


add_fake_nodes = False
if add_fake_nodes:
    cell_type_counts = nf.nodes["cell_type"].value_counts()

    label_feature = "column_id"
    cell_to_label_counts = (
        nf.nodes.pivot_table(index="cell_type", columns=label_feature, aggfunc="size")
        .fillna(0)
        .astype(int)
    )

    i = 1
    fake_nodes = []
    missing_cell_ilocs = np.argwhere(cell_to_label_counts == 0)
    for cell_type_iloc, column_id_iloc in missing_cell_ilocs:
        cell_type = cell_to_label_counts.index[cell_type_iloc]
        column_id = cell_to_label_counts.columns[column_id_iloc]
        fake_nodes.append(
            {
                "cell_id": -i,
                "column_id": column_id,
                "cell_type": cell_type,
            }
        )
        i += 1

    fake_nodes = pd.DataFrame(fake_nodes)
    fake_nodes.set_index("cell_id", inplace=True)
    fake_nodes

    nf.add_nodes(fake_nodes, inplace=True)

    nf.nodes

# %%

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)
nf.nodes

# %%

A = nf.to_adjacency(weight_col="weight").values.astype(float)

# %%

# this is a matrix to match to
# it has block diagonals of shape 31 x 31
# matching to this matrix tries to force an alignment where edges are within a column
dummy_labels = np.concatenate([np.full(n_types, i) for i in range(1, n_columns + 1)])
mask = (dummy_labels[:, None] == dummy_labels[None, :]).astype(float)
B = mask.astype(float)

# %%
uni_cell_types = np.unique(nf.nodes["cell_type"])
dummy_cell_types = np.concatenate([uni_cell_types for i in range(n_columns)])
target_df = pd.DataFrame(
    {
        "cell_type": dummy_cell_types,
        "column_id": dummy_labels,
    }
)
target_df

# %%
S = pd.DataFrame(index=nf.nodes.index, columns=target_df.index).fillna(0)
S = nf.nodes["cell_type"].values[:, None] == target_df["cell_type"].values[None, :]
S = S.astype(float)

# %%

# this is a matrix keeping track of the cell type labels
# it has 1s where the cell types are the same
# we'll try to also match the negative of this to the matrix B above
# which has the effect of trying to minimize cell types being in the same column
# cell_type_labels = nf.nodes["cell_type"].values
# mask = cell_type_labels[:, None] == cell_type_labels[None, :]
# mask[np.diag_indices_from(mask)] = False
# S = mask.astype(float)
# S = -S
# S = csr_array(S)


# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.heatmap(
    A > 0,
    ax=axs[0],
    cmap="RdBu_r",
    center=0,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
    square=True,
)
axs[0].set_title("A")

sns.heatmap(
    B,
    ax=axs[1],
    cmap="RdBu_r",
    center=0,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
    square=True,
)

axs[1].set_title("B")

sns.heatmap(
    S,
    ax=axs[2],
    cmap="RdBu_r",
    center=0,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
    square=True,
)
axs[2].set_title("S")

# %%

# 545.128 seconds for one iteration on full sized

# A_mod = A.copy()
# A_mod.data += np.random.normal(0, 1, A_mod.data.shape[0])


currtime = time.time()
result = graph_match(
    A,
    B,
    S=S * 100,
    max_iter=60,
    shuffle_input=False,
    n_init=50,
    # tol=0.00001,
    # init_perturbation=0.001,
    verbose=10,
    fast=True,
    n_jobs=8,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
result.misc


# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.heatmap(
    A[result.indices_A][:, result.indices_A] > 0,
    ax=axs[0],
    cbar=False,
    square=True,
    cmap="RdBu_r",
    center=0,
)

sns.heatmap(
    B[result.indices_B][:, result.indices_B],
    ax=axs[1],
    cbar=False,
    square=True,
    cmap="RdBu_r",
    center=0,
)

sns.heatmap(
    S[result.indices_A][:, result.indices_B],
    ax=axs[2],
    cbar=False,
    square=True,
    cmap="RdBu_r",
    center=0,
)


# %%

indices_A = result.indices_A
indices_B = result.indices_B
# predicted_labels = dummy_labels[indices_B]
# nf.nodes["predicted_labels"] = predicted_labels

reordered_index = nf.nodes.index[indices_A]

predicted_labels = pd.Series(index=reordered_index, data=dummy_labels[indices_B])

nf.nodes["predicted_labels"] = predicted_labels
# %%
random_labels = nf.nodes["column_id"].copy()
random_labels = random_labels.sample(frac=1).values
nf.nodes["random_labels"] = random_labels

# %%

label_feature = "predicted_labels"


def compute_metrics(nf, label_feature):
    edges = nf.apply_node_features(label_feature).edges
    n_within_group = edges.query(f"source_{label_feature} == target_{label_feature}")[
        "weight"
    ].sum()

    nodes = nf.nodes
    # don't count the dummy nodes here
    nodes = nodes[nodes.index > 0]
    cell_to_label_counts = (
        nodes.pivot_table(index="cell_type", columns=label_feature, aggfunc="size")
        .fillna(0)
        .astype(int)
    )

    violations_mask = cell_to_label_counts > 1
    violations = int(cell_to_label_counts[violations_mask].sum().sum())

    return n_within_group, cell_to_label_counts, violations


old_n_within_group, old_cell_to_label_counts, old_violations = compute_metrics(
    nf, "column_id"
)
print("Old:")
print(old_n_within_group)
print(old_violations)
print()

new_n_within_group, new_cell_to_label_counts, new_violations = compute_metrics(
    nf, "predicted_labels"
)
print("New:")
print(new_n_within_group)
print(new_violations)
print()

random_n_within_group, random_cell_to_label_counts, random_violations = compute_metrics(
    nf, "random_labels"
)
print("Random:")
print(random_n_within_group)
print(random_violations)
print()

nodes = nf.nodes[nf.nodes.index > 0]
ari = adjusted_rand_score(nodes["predicted_labels"].values, nodes["column_id"].values)
print(f"ARI: {ari:.2f}")

# %%
