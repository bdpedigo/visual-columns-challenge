# %%
import pickle

import numpy as np
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

nf = load_networkframe(
    path="visual-columns-challenge/results/submissions/columns_v3.csv"
)

label_feature = "column_id"

add_fake_nodes(nf, label_feature)
target_nodes = create_target_nodes(nf)
B = create_matching_target(target_nodes)

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)

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
# reload_name = None
# reload_from = "test=False-class_weight=75-tol=0.001-max_iter=100-sparse=True"
# reload_from = "1710535591"
# reload_from = "class_weight=75-max_iter=100-restart=True"
# reload_from = "1710544091"
reload_from = "1712548568"
reload = reload_from is not None
reload_from_iter = 159

if reload_from is not None:
    load_path = OUT_PATH
    if "=" in reload_from:
        load_path = load_path / "old"
    else:
        load_path = load_path / reload_from
    load_path = load_path / f"{reload_from}_results_by_iter.pkl"
    with open(load_path, "rb") as f:
        results_by_iter = pickle.load(f)

        result = results_by_iter[reload_from_iter]

    indices_A = result.indices_A
    indices_B = result.indices_B

n = len(indices_A)
# %%
# experiment_params = {
#     "max_iter": max_iter,
#     "class_weight": class_weight,
#     "tol": tol,
#     "reload_from": reload_from,
#     "reload_from_iter": reload_from_iter,
#     "reload": reload,
# }
# experiment_id = int(time.time())
# save_name = str(experiment_id)

# experiment_df = pd.Series(experiment_params, name=experiment_id).to_frame().T

# try:
#     manifest = pd.read_csv(OUT_PATH / "experiment_manifest.csv", index_col=0)
# except pd.errors.EmptyDataError:
#     manifest = pd.DataFrame()
# manifest = pd.concat([manifest, experiment_df], axis=0)
# manifest.index.name = "experiment_id"
# manifest.to_csv(OUT_PATH / "experiment_manifest.csv")

results_by_iter = []
scores = []
last_solution = np.eye(n)

if reload_from is not None:
    last_solution = last_solution[indices_A][:, indices_B]
    last_perm = indices_B
else:
    from graspologic.match.wrappers import MatchResult

    last_perm = np.arange(n)

    result = MatchResult(
        indices_A=last_perm,
        indices_B=last_perm,
        score=0.0,
        mist=[{}],
    )


# TODO add something here for computing score from the initial solution and storing for
# iteration "0"


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

print("New:")
print(corrected_n_within_group)
print(corrected_n_matched)
print(corrected_violations)

# iter_scores = {
#     "experiment_id": experiment_id,
#     "n_within_group": new_n_within_group,
#     "n_matched": new_n_matched,
#     "n_violations": new_violations,
#     "corrected_n_within_group": corrected_n_within_group,
#     "corrected_n_matched": corrected_n_matched,
#     "corrected_violations": corrected_violations,
#     "only_real": only_real,
#     "only_target": only_target,
#     "swaps_from_last": 0,
#     "iteration": 0,
#     "time": 0,
#     "nnz": np.count_nonzero(last_solution),
# }
# scores.append(iter_scores)

# %%
corrected_nf.query_nodes('node_type == "real"')
submission_nodes = corrected_nf.nodes[["cell_type", "predicted_column_id"]].copy()
submission_nodes.reset_index(inplace=True)
submission_nodes.rename(
    columns={
        "cell_id": "cell id",
        "cell_type": "cell type",
        "predicted_column_id": "column id",
    },
    inplace=True,
)

import pandas as pd
from pkg import DATA_PATH

original_nodes = pd.read_csv(DATA_PATH / "ol_columns.csv")

missing = ~original_nodes["cell id"].isin(submission_nodes["cell id"])

missing_nodes = pd.DataFrame(original_nodes[missing])
missing_nodes["column id"] = np.nan
missing_nodes["column id"] = missing_nodes["column id"].astype("Int64")

submission_nodes = pd.concat([submission_nodes, missing_nodes], ignore_index=True)
submission_nodes = submission_nodes.sort_values(["cell id"]).reset_index(drop=True)
submission_nodes["column id"] = (
    submission_nodes["column id"].astype("object").fillna("not assigned")
)
submission_nodes.to_csv(OUT_PATH / "submissions" / "columns_v4.csv", index=False)

# %%
test_nf = load_networkframe(path=OUT_PATH / "submissions" / "columns_v4.csv")

test_nf = test_nf.query_nodes('node_type == "real" & column_id.notna()').copy()
test_n_within_group, test_n_matched, test_violations = compute_metrics(
    test_nf, "column_id"
)
print("New test:")
print(test_n_within_group)
print(test_n_matched)
print(test_violations)
print()

# %%
