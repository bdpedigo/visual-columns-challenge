# TODO could iteratively refine the actual subgraph that one is looking for, i.e. make
# the block diagonal equal to the average one of these column subgraphs
# i doubt this will do better on the actual problem, but it could be interesting!


# %%
import os
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
)
from scipy.sparse import csr_array

from graspologic.match import graph_match

load_path = "visual-columns-challenge/results/submissions/BP_columns_v1.csv"
nf = load_networkframe(
    path=load_path,
)
# %%

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

# only reward matching nodes of the same cell type
node_cell_type = nf.nodes["cell_type"].values
target_cell_type = target_nodes["cell_type"].values
S = node_cell_type[:, None] == target_cell_type[None, :]

only_real = False
if only_real:
    # only reward matching a real node
    node_mask = nf.nodes["node_type"].values == "real"
    S = node_mask[:, None] & S

only_target = False
if only_target:
    # only reward matching to an actual target node
    target_mask = target_nodes["node_type"].values == "target"
    S = S & target_mask[None, :]

S = S.astype(float)


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

max_iter = 100
class_weight = 200  # 150
tol = 0.001
sparse_position = True

damping_factor = None

A_input = csr_array(A)
B_input = csr_array(B)
S_input = csr_array(S)

# reload_from = "class_weight=75-max_iter=100-restart=True"
# reload_from_iter = 67

results_by_iter = []
scores = []

# reload_from = "1712153817"
# reload_from_iter = 45

# reload_from                 1712211078
# reload_from_iter                    90

# reload_from = "1712247272"
# reload_from_iter = 18
# 1712528432
reload_from = None
reload_convex = False
reload = reload_from is not None
reload_from_iter = -1

reload_from = "1712528432"
reload_from_iter = 16
reload_convex = True


if reload_from is not None:
    load_path = OUT_PATH
    if "=" in reload_from:
        load_path = load_path / "old"
    else:
        load_path = load_path / reload_from
    load_path = load_path / f"{reload_from}_results_by_iter.pkl"
    with open(load_path, "rb") as f:
        results_by_iter = pickle.load(f)
        print("reloaded")
        result = results_by_iter[reload_from_iter - 1]

    indices_A = result.indices_A
    indices_B = result.indices_B

    if reload_convex:
        last_solution = result.misc[0]["convex_solution"].toarray()
    else:
        last_solution = np.eye(A_input.shape[0])
        last_solution = last_solution[indices_A][:, indices_B]
    last_perm = indices_B
else:
    last_solution = np.eye(A_input.shape[0])
    last_perm = np.arange(B_input.shape[0])

    from graspologic.match.wrappers import MatchResult

    result = MatchResult(
        indices_A=last_perm,
        indices_B=last_perm,
        score=0.0,
        misc=[{}],
    )

label_masking = False
if label_masking:
    labels = target_nodes["cell_type"].values
else:
    labels = None

# %%

perturb_steps = 20
perturb_weight = 0.01

experiment_params = {
    "max_iter": max_iter,
    "class_weight": class_weight,
    "tol": tol,
    "reload_from": reload_from,
    "reload_from_iter": reload_from_iter,
    "reload": reload,
    "only_real": only_real,
    "only_target": only_target,
    "reload_convex": reload_convex,
    "sparse_position": sparse_position,
    "damping_factor": damping_factor,
    "label_masking": label_masking,
    "load_path": load_path,
    "perturb_steps": perturb_steps,
    "perturb_weight": perturb_weight,
}
experiment_id = int(time.time())
save_name = str(experiment_id)

experiment_df = pd.Series(experiment_params, name=experiment_id).to_frame().T

try:
    manifest = pd.read_csv(OUT_PATH / "experiment_manifest.csv", index_col=0)
except pd.errors.EmptyDataError:
    manifest = pd.DataFrame()
manifest = pd.concat([manifest, experiment_df], axis=0)
manifest.index.name = "experiment_id"
manifest.to_csv(OUT_PATH / "experiment_manifest.csv")


out_path = OUT_PATH / save_name

if not os.path.exists(out_path):
    os.makedirs(out_path)


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
    "experiment_id": experiment_id,
    "n_within_group": new_n_within_group,
    "n_matched": new_n_matched,
    "n_violations": new_violations,
    "corrected_n_within_group": corrected_n_within_group,
    "corrected_n_matched": corrected_n_matched,
    "corrected_violations": corrected_violations,
    "swaps_from_last": 0,
    "iteration": 0,
    "time": 0,
    "score": result.score,
    "nnz": np.count_nonzero(last_solution),
    **experiment_params,
}
scores.append(iter_scores)

print()
print("Initial scores:")
print(iter_scores)
print()

bary = np.full((A_input.shape[0], B_input.shape[0]), 1 / B_input.shape[0])

n_init = 1
all_time = time.time()
for i in range(1, max_iter + 1):
    print("Iteration:", i)
    currtime = time.time()
    if isinstance(last_solution, csr_array):
        last_solution = last_solution.toarray()
    if i % perturb_steps == 0:
        last_solution = (1 - perturb_weight) * last_solution + perturb_weight * bary
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
        sparse_position=sparse_position,
        damping_factor=damping_factor,
        labels=labels,
    )
    assert (result.indices_A == np.arange(A_input.shape[0])).all()
    solve_time = time.time() - currtime
    print(f"{solve_time:.3f} seconds elapsed to solve")

    last_solution = result.misc[0]["convex_solution"]
    sparse_soln = csr_array(result.misc[0]["convex_solution"])
    nnz = sparse_soln.nnz

    swaps = (last_perm != result.indices_B).sum()

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
        "experiment_id": experiment_id,
        "n_within_group": new_n_within_group,
        "n_matched": new_n_matched,
        "n_violations": new_violations,
        "corrected_n_within_group": corrected_n_within_group,
        "corrected_n_matched": corrected_n_matched,
        "corrected_violations": corrected_violations,
        "only_real": only_real,
        "only_target": only_target,
        "swaps_from_last": swaps,
        "iteration": i,
        "time": solve_time,
        "score": result.score,
        "nnz": nnz,
        **experiment_params,
    }
    scores.append(iter_scores)
    print()
    print("Iteration scores:")
    print(iter_scores)
    print()

    result.misc[0]["convex_solution"] = csr_array(result.misc[0]["convex_solution"])
    results_by_iter.append(result)

print()
with open(out_path / f"{save_name}_final_result.pkl", "wb") as f:
    # result.misc[0]["convex_solution"] = None
    pickle.dump(result, f)

with open(out_path / f"{save_name}_results_by_iter.pkl", "wb") as f:
    pickle.dump(results_by_iter, f)

matched_nf.nodes.to_csv(out_path / f"{save_name}-matched_nodes.csv")
corrected_nf.nodes.to_csv(out_path / f"{save_name}-corrected_nodes.csv")
target_nodes.to_csv(out_path / f"{save_name}-target_nodes.csv")

print("\n---")
print(f"{time.time() - all_time:.3f} seconds elapsed in total")
print("---\n")

# %%
