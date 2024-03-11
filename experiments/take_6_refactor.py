# TODO could iteratively refine the actual subgraph that one is looking for, i.e. make
# the block diagonal equal to the average one of these column subgraphs
# i doubt this will do better on the actual problem, but it could be interesting!


# %%
import pickle
import time

import numpy as np
import pandas as pd
from pkg import (
    OUT_PATH,
    add_fake_nodes,
    compute_metrics,
    correct_violations,
    create_matched_networkframe,
    create_matching_target,
    create_target_nodes,
    load_networkframe,
    plot_matched_matrices,
)
from scipy.sparse import csr_array

from graspologic.match import graph_match

test = False
if test:
    nf = load_networkframe(sample=10)
else:
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
    fig, axs = plot_matched_matrices(A, B, S)

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
tol = 0.001

A_input = csr_array(A)
B_input = csr_array(B)
S_input = csr_array(S)

# reload_name = None
reload_from = "test=False-class_weight=75-tol=0.001-max_iter=100-sparse=True"
reload = reload_from is not None
reload_from_iter = -1

if reload_from is not None:
    with open(OUT_PATH / f"{reload_from}_results_by_iter.pkl", "rb") as f:
        results_by_iter = pickle.load(f)

        result = results_by_iter[reload_from_iter]

    indices_A = result.indices_A
    indices_B = result.indices_B
# %%
experiment_params = {
    "max_iter": max_iter,
    "class_weight": class_weight,
    "tol": tol,
    "reload_from": reload_from,
    "reload_from_iter": reload_from_iter,
    "reload": reload,
}
experiment_id = int(time.time())
experiment_df = pd.Series(experiment_params, name=experiment_id).to_frame().T

try:
    manifest = pd.read_csv(OUT_PATH / "experiment_manifest.csv", index_col=0)
except pd.errors.EmptyDataError:
    manifest = pd.DataFrame()
manifest = pd.concat([manifest, experiment_df], axis=0)
manifest.index.name = "experiment_id"
manifest.to_csv(OUT_PATH / "experiment_manifest.csv")

# %%

save_name = str(experiment_id)

results_by_iter = []
scores = []
last_solution = np.eye(A_input.shape[0])

if reload_from is not None:
    last_solution = last_solution[indices_A][:, indices_B]
    last_perm = indices_B
else:
    last_perm = np.arange(B_input.shape[0])

n_init = 1
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
    sparse_soln = csr_array(result.misc[0]["convex_solution"])
    nnz = sparse_soln.nnz
    print(f"Solution nnz: {nnz}")

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

    if not test:
        score_df = pd.DataFrame(scores)
        score_df.to_csv(OUT_PATH / f"{save_name}_scores.csv")

    if stable_step_counter >= max_stable_steps:
        print("Converged!")
        break

if not test:
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
    plot_matched_matrices(A, B, S, result)
