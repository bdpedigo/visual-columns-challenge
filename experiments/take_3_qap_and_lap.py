# %%
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from scipy.sparse import csr_array

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

np.random.seed(8888)

n_columns = 796
test = False
if test:
    n_select_columns = 10
    select_cols = np.random.choice(
        np.arange(1, n_columns + 1), size=n_select_columns, replace=False
    )
    # just for testing
    unassigned_nodes = (
        nf.nodes.query("column_id.isna()")
        .index.to_series()
        .sample(frac=n_select_columns / n_columns)
    )
    nf = nf.query_nodes(
        "column_id.isin(@select_cols) | index.isin(@unassigned_nodes)",
        local_dict=locals(),
    ).copy()

    n_columns = n_select_columns

n_types = 31
n_ideal_cells = n_columns * n_types

# %%
nf.nodes["node_type"] = "real"

# %%
len(np.unique(nf.edges[["source", "target"]].values, axis=0))

# %%
len(nf.edges)

# %%
label_feature = "column_id"
edges = nf.apply_node_features(label_feature).edges
edges = edges.query("source_column_id.notna() & target_column_id.notna()")
edges = edges.query("source != target")
n_within_group = edges.query(f"source_{label_feature} == target_{label_feature}")[
    "weight"
].sum()
print(n_within_group)
print(1_381_915)
# %%

label_feature = "column_id"
cell_to_label_counts = (
    nf.nodes.pivot_table(
        index="cell_type", columns=label_feature, aggfunc="size", dropna=False
    )
    .fillna(0)
    .astype(int)
)
cell_to_label_counts

# %%

cell_to_label_counts.sum(axis=1)

# if there are fewer than n_columns of a given cell type, then we should make some fake
# nodes in the nodes dataframe to represent these
# if there are more than n_columns of a given cell type, then we should also make
# some fake nodes in the target dataframe to represent these
# ...
# i think?

# %%

# TODO look into sweeping over the weighting of AB vs S terms

# TODO FAQ-2-opt: do some kind of optimize, perturb, optimize, perturb scheme
# should be easy to implement with an outer loop and a literal permutation of the input
# myself

# TODO look into an adopted padding version of this? could that matter here

# TODO could iteratively refine the actual subgraph that one is looking for, i.e. make
# the block diagonal equal to the average one of these column subgraphs
# i doubt this will do better on the actual problem, but it could be interesting!

cell_type_counts = nf.nodes["cell_type"].value_counts()

label_feature = "column_id"
cell_to_label_counts = (
    nf.nodes.pivot_table(index="cell_type", columns=label_feature, aggfunc="size")
    .fillna(0)
    .astype(int)
)

add_fake_nodes = True
if add_fake_nodes:
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
                "node_type": "fake",
            }
        )
        i += 1

    fake_nodes = pd.DataFrame(fake_nodes)
    fake_nodes.set_index("cell_id", inplace=True)
    fake_nodes

    nf.add_nodes(fake_nodes, inplace=True)

nf.nodes
# %%
cell_to_label_counts = (
    nf.nodes.pivot_table(
        index="cell_type", columns=label_feature, aggfunc="size", dropna=False
    )
    .fillna(0)
    .astype(int)
)
cell_type_counts = cell_to_label_counts.sum(axis=1)
cell_type_counts

# %%
dummy_labels = np.concatenate([np.full(n_types, i) for i in range(1, n_columns + 1)])
uni_cell_types = np.unique(nf.nodes["cell_type"])
dummy_cell_types = np.concatenate([uni_cell_types for _ in range(n_columns)])
target_nodes = pd.DataFrame(
    {
        "cell_type": dummy_cell_types,
        "column_id": dummy_labels,
        "node_type": "target",
    }
)
target_nodes
# %%
cell_type_extras = cell_type_counts - n_columns
cell_type_extras
# %%
extra_nodes = pd.DataFrame(
    {
        "cell_type": np.repeat(cell_type_extras.index, cell_type_extras),
        "column_id": 9999,
        "node_type": "extra",
    }
)
extra_nodes

# %%
target_nodes = pd.concat([target_nodes, extra_nodes], ignore_index=True)
target_nodes.sort_values(["column_id", "cell_type"], inplace=True)
# %%
# these are "fake" edges

target_labels = target_nodes["column_id"].values
mask = (target_labels[:, None] == target_labels[None, :]).astype(float)
not_fake = (target_nodes["column_id"] != 9999).values
not_fake_mask = not_fake[:, None] & not_fake[None, :]
B = (mask * not_fake_mask).astype(float)

# %%

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)
nf.nodes

# %%

A = nf.to_adjacency(weight_col="weight").values.astype(float)

# %%

# # this is a matrix to match to
# # it has block diagonals of shape 31 x 31
# # matching to this matrix tries to force an alignment where edges are within a column
# dummy_labels = np.concatenate([np.full(n_types, i) for i in range(1, n_columns + 1)])
# mask = (dummy_labels[:, None] == dummy_labels[None, :]).astype(float)
# B = mask.astype(float)

# %%


# %%

S = pd.DataFrame(index=nf.nodes.index, columns=target_nodes.index, dtype=float).fillna(
    0.0
)
S = nf.nodes["cell_type"].values[:, None] == target_nodes["cell_type"].values[None, :]
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

if test:
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

max_iter = 30
class_weight = 100
n_init = 1
tol = 0.001
sparse = True

if sparse:
    A_input = csr_array(A)
    B_input = csr_array(B)
    S_input = csr_array(S)
else:
    A_input = A
    B_input = B
    S_input = S

currtime = time.time()
result = graph_match(
    A_input,
    B_input,
    S=S_input * class_weight,
    max_iter=max_iter,
    shuffle_input=False,
    n_init=n_init,
    tol=tol,
    # init_perturbation=0.001,
    verbose=10,
    fast=True,
    n_jobs=1,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
print()

# %%
result.misc

# %%
if test:
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

reordered_node_index = nf.nodes.index[indices_A]

reordered_target_index = target_nodes.index[indices_B]
reordered_targets = target_nodes.loc[reordered_target_index].copy()

reordered_targets.index = reordered_node_index
reordered_targets.rename(columns=lambda x: f"predicted_{x}", inplace=True)

matched_nodes = nf.nodes.join(reordered_targets)

matched_nf = NetworkFrame(
    nodes=matched_nodes,
    edges=nf.edges,
)

# %%
benchmark_nf = nf.query_nodes('node_type == "real" & column_id.notna()').copy()

# %%
matched_nf.query_nodes(
    "node_type == 'real' & predicted_node_type == 'target'", inplace=True
)

# %%
# TODO write some code to post-process and remove nodes that are improperly matched
# or, increase the weighting of that term in the objective function

# %%


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

    n_matched = len(nf)

    return n_within_group, n_matched, violations


old_n_within_group, old_n_matched, old_violations = compute_metrics(
    benchmark_nf, "column_id"
)
print("Old:")
print(old_n_within_group)
print(old_violations)
print()

new_n_within_group, new_n_matched, new_violations = compute_metrics(
    matched_nf, "predicted_column_id"
)
print("New:")
print(new_n_within_group)
print(new_violations)
print()

# %%

save_name = f"test={test}-n_columns={n_columns}-fake_nodes={add_fake_nodes}-class_weight={class_weight}-n_init={n_init}-tol={tol}-max_iter={max_iter}-sparse={sparse}"

with open(f"{save_name}.bin", "wb") as f:
    result.misc[0]["convex_solution"] = None
    pickle.dump(result, f)

matched_nf.nodes.to_csv(f"{save_name}-matched_nodes.csv")
target_nodes.to_csv(f"{save_name}-target_nodes.csv")

# %%
file = "test=False-n_columns=796-fake_nodes=True-class_weight=100-n_init=1-tol=0.001-max_iter=30-sparse=True.bin"
with open(file, "rb") as f:
    result = pickle.load(f)

# %%

changes = result.misc[0]["changes"]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(x=np.arange(len(changes)), y=changes, ax=axs[0])
sns.lineplot(x=np.arange(len(changes)), y=changes, ax=axs[1])
axs[1].set_yscale("log")