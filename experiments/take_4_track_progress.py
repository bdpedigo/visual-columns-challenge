# TODO look into sweeping over the weighting of AB vs S terms

# TODO FAQ-2-opt: do some kind of optimize, perturb, optimize, perturb scheme
# should be easy to implement with an outer loop and a literal permutation of the input
# myself

# TODO look into an adopted padding version of this? could that matter here

# TODO could iteratively refine the actual subgraph that one is looking for, i.e. make
# the block diagonal equal to the average one of these column subgraphs
# i doubt this will do better on the actual problem, but it could be interesting!

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
test = True
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

nf.nodes["node_type"] = "real"


# %%
def cell_to_column_crosstab(nodes, label_feature="column_id", dropna=False):
    out = (
        nodes.pivot_table(
            index="cell_type", columns=label_feature, aggfunc="size", dropna=dropna
        )
        .fillna(0)
        .astype(int)
    )
    return out


label_feature = "column_id"
cell_to_label_counts = cell_to_column_crosstab(nf.nodes, label_feature)


# %%

cell_type_counts = nf.nodes["cell_type"].value_counts()

cell_to_label_counts = cell_to_column_crosstab(nf.nodes, label_feature, dropna=True)

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

cell_type_counts = nf.nodes["cell_type"].value_counts().sort_index()
cell_type_extras = cell_type_counts - n_columns

extra_nodes = pd.DataFrame(
    {
        "cell_type": np.repeat(cell_type_extras.index, cell_type_extras),
        "column_id": np.nan,
        "node_type": "extra",
    }
)

target_nodes = pd.concat([target_nodes, extra_nodes], ignore_index=True)
target_nodes["column_id"] = target_nodes["column_id"].astype("Int64")
target_nodes.sort_values(["column_id", "cell_type"], inplace=True)
# %%
# these are "fake" edges

target_labels = target_nodes["column_id"].fillna(-1).values
mask = (target_labels[:, None] == target_labels[None, :]).astype(float)
not_fake = (target_nodes["column_id"].notna()).values
not_fake_mask = not_fake[:, None] & not_fake[None, :]
B = (mask * not_fake_mask).astype(float)

# %%

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)
nf.nodes

# %%

A = nf.to_adjacency(weight_col="weight").values.astype(float)


# %%

S = pd.DataFrame(index=nf.nodes.index, columns=target_nodes.index, dtype=float).fillna(
    0.0
)
S = nf.nodes["cell_type"].values[:, None] == target_nodes["cell_type"].values[None, :]
S = S.astype(float)


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


def create_matched_networkframe(nf, result):
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
    matched_nf.query_nodes(
        "node_type == 'real' & predicted_node_type == 'target'", inplace=True
    )
    return matched_nf


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


benchmark_nf = nf.query_nodes('node_type == "real" & column_id.notna()').copy()
old_n_within_group, old_n_matched, old_violations = compute_metrics(
    benchmark_nf, "column_id"
)
print("Old:")
print(old_n_within_group)
print(old_n_matched)
print(old_violations)
print()


max_iter = 100
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

results_by_iter = []
scores = []
last_solution = np.eye(A_input.shape[0])
last_perm = np.arange(B_input.shape[0])
max_stable_steps = 5
stable_step_counter = 0
for i in range(max_iter):
    print("Iteration:", i)
    currtime = time.time()
    result = graph_match(
        A_input,
        B_input,
        S=S_input * class_weight,
        max_iter=1,
        shuffle_input=False,
        n_init=n_init,
        init=last_solution,
        tol=tol,
        # init_perturbation=0.001,
        verbose=5,
        fast=True,
        n_jobs=1,
    )
    print(f"{time.time() - currtime:.3f} seconds elapsed to solve")

    last_solution = result.misc[0]["convex_solution"]

    swaps = (last_perm != result.indices_B).sum()
    last_perm = result.indices_B
    print(f"Swaps: {swaps}")

    matched_nf = create_matched_networkframe(nf, result)

    new_n_within_group, new_n_matched, new_violations = compute_metrics(
        matched_nf, "predicted_column_id"
    )
    scores.append(
        {
            "n_within_group": new_n_within_group,
            "n_matched": new_n_matched,
            "n_violations": new_violations,
            "swaps_from_last": swaps,
        }
    )
    print("New scores:")
    print(f"Synapses in columns: {new_n_within_group}")
    print(f"Matched nodes: {new_n_matched}")
    print(f"Violations: {new_violations}")
    print()

    result.misc[0]["convex_solution"] = None
    results_by_iter.append(result)


save_name = f"test={test}-n_columns={n_columns}-fake_nodes={add_fake_nodes}-class_weight={class_weight}-n_init={n_init}-tol={tol}-max_iter={max_iter}-sparse={sparse}"

with open(f"{save_name}_final_result.pkl", "wb") as f:
    result.misc[0]["convex_solution"] = None
    pickle.dump(result, f)

with open(f"{save_name}_results_by_iter.pkl", "wb") as f:
    pickle.dump(results_by_iter, f)

matched_nf.nodes.to_csv(f"{save_name}-matched_nodes.csv")
target_nodes.to_csv(f"{save_name}-target_nodes.csv")


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
# TODO write some code to post-process and remove nodes that are improperly matched
# or, increase the weighting of that term in the objective function


# %%


# %%
# file = "test=False-n_columns=796-fake_nodes=True-class_weight=100-n_init=1-tol=0.001-max_iter=30-sparse=True.bin"
# with open(file, "rb") as f:
#     result = pickle.load(f)

# # %%

# changes = result.misc[0]["changes"]

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# sns.lineplot(x=np.arange(len(changes)), y=changes, ax=axs[0])
# sns.lineplot(x=np.arange(len(changes)), y=changes, ax=axs[1])
# axs[1].set_yscale("log")
