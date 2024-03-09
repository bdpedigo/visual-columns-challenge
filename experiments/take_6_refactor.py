# TODO FAQ-2-opt: do some kind of optimize, perturb, optimize, perturb scheme
# should be easy to implement with an outer loop and a literal permutation of the input
# myself

# TODO look into an adopted padding version of this? could that matter here

# TODO could iteratively refine the actual subgraph that one is looking for, i.e. make
# the block diagonal equal to the average one of these column subgraphs
# i doubt this will do better on the actual problem, but it could be interesting!

# TODO consider a version of this where there is a penalty for not matching a node?
# or is that already implicit somehow?

# TODO consider adding a regularization barycentric term to the solutions, or maybe
# only doing so when one gets stuck, or stuck in a cycle

# %%
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg import (
    OUT_PATH,
    add_fake_nodes,
    compute_metrics,
    correct_violations,
    create_matched_networkframe,
    create_matching_target,
    create_target_nodes,
    load_networkframe,
)
from scipy.sparse import csr_array

from graspologic.match import graph_match

test = False
nf = load_networkframe()

label_feature = "column_id"

add_fake_nodes(nf, label_feature)
target_nodes = create_target_nodes(nf)
B = create_matching_target(target_nodes)

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)

# %%

A = nf.to_adjacency(weight_col="weight").values.astype(float)

# %%

S = pd.DataFrame(index=nf.nodes.index, columns=target_nodes.index, dtype=float).fillna(
    0.0
)

# %%
only_real_constraint = False
if only_real_constraint:
    # only want to reward matching a real node to a target node
    node_mask = nf.nodes["node_type"].values == "real"
    target_mask = target_nodes["node_type"].values == "target"
    mask = node_mask[:, None] & target_mask[None, :]

    # only want to reward matching nodes of the same cell type
    node_cell_type = nf.nodes["cell_type"].values
    target_cell_type = target_nodes["cell_type"].values
    matching_cell_types = node_cell_type[:, None] == target_cell_type[None, :]

    S = mask & matching_cell_types

else:
    S = (
        nf.nodes["cell_type"].values[:, None]
        == target_nodes["cell_type"].values[None, :]
    )
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


benchmark_nf = nf.query_nodes('node_type == "real" & column_id.notna()').copy()
old_n_within_group, old_n_matched, old_violations = compute_metrics(
    benchmark_nf, "column_id"
)
print("Old:")
print(old_n_within_group)
print(old_n_matched)
print(old_violations)
print()

# %%

max_iter = 1
class_weight = 75  # 150
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

reload_name = "test=False-class_weight=75-tol=0.001-max_iter=100-sparse=True"
reload_name = None

if reload_name is not None:
    with open(OUT_PATH / f"{reload_name}_final_result.pkl", "rb") as f:
        result = pickle.load(f)

    indices_A = result.indices_A
    indices_B = result.indices_B

# %%

save_name = (
    f"class_weight={class_weight}-max_iter={max_iter}-restart={reload_name is not None}"
)

results_by_iter = []
scores = []
last_solution = np.eye(A_input.shape[0])

if reload_name is not None:
    last_solution = last_solution[indices_A][:, indices_B]
    last_perm = indices_B
else:
    last_perm = np.arange(B_input.shape[0])

max_stable_steps = 5
stable_step_counter = 0
all_time = time.time()
for i in range(1, max_iter + 1):
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
    solve_time = time.time() - currtime
    print(f"{solve_time:.3f} seconds elapsed to solve")

    last_solution = result.misc[0]["convex_solution"]

    swaps = (last_perm != result.indices_B).sum()
    last_perm = result.indices_B
    print(f"Swaps: {swaps}")
    if swaps == 0:
        stable_step_counter += 1
    else:
        stable_step_counter = 0

    matched_nf = create_matched_networkframe(nf, result, target_nodes)

    new_n_within_group, new_n_matched, new_violations = compute_metrics(
        matched_nf, "predicted_column_id"
    )

    corrected_nf = correct_violations(matched_nf, "predicted_column_id")

    (
        corrected_n_within_group,
        corrected_n_matched,
        corrected_violations,
    ) = compute_metrics(corrected_nf, "predicted_column_id")

    iter_scores = {
        "n_within_group": new_n_within_group,
        "n_matched": new_n_matched,
        "n_violations": new_violations,
        "corrected_n_within_group": corrected_n_within_group,
        "corrected_n_matched": corrected_n_matched,
        "corrected_violations": corrected_violations,
        "swaps_from_last": swaps,
        "iteration": i,
        "time": solve_time,
    }
    scores.append(iter_scores)
    print("Iteration scores:")
    print(iter_scores)
    print()

    # result.misc[0]["convex_solution"] = None
    results_by_iter.append(result)

    score_df = pd.DataFrame(scores)
    score_df.to_csv(OUT_PATH / f"{save_name}_scores.csv")

    if stable_step_counter >= max_stable_steps:
        print("Converged!")
        break


with open(OUT_PATH / f"{save_name}_final_result.pkl", "wb") as f:
    # result.misc[0]["convex_solution"] = None
    pickle.dump(result, f)

with open(OUT_PATH / f"{save_name}_results_by_iter.pkl", "wb") as f:
    pickle.dump(results_by_iter, f)


matched_nf.nodes.to_csv(OUT_PATH / f"{save_name}-matched_nodes.csv")
corrected_nf.nodes.to_csv(OUT_PATH / f"{save_name}-corrected_nodes.csv")
target_nodes.to_csv(OUT_PATH / f"{save_name}-target_nodes.csv")

print("\n---")
print(f"{time.time() - all_time:.3f} seconds elapsed in total")
print("---\n")


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
