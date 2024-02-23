# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from networkframe import NetworkFrame
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

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
test = False
if test:
    n_select_columns = 20
    select_cols = np.random.choice(
        np.arange(1, n_columns + 1), size=n_select_columns, replace=False
    )
    # just for testing
    nf = nf.query_nodes("column_id.isin(@select_cols)", local_dict=locals()).copy()
    n_columns = n_select_columns

n_types = 31
n_ideal_cells = n_columns * n_types

# %%
cell_type_counts = nf.nodes["cell_type"].value_counts()

label_feature = "column_id"
cell_to_label_counts = (
    nf.nodes.pivot_table(index="cell_type", columns=label_feature, aggfunc="size")
    .fillna(0)
    .astype(int)
)
cell_to_label_counts
# %%
# i = 1
# fake_nodes = []
# missing_cell_ilocs = np.argwhere(cell_to_label_counts == 0)
# for cell_type_iloc, column_id_iloc in missing_cell_ilocs:
#     cell_type = cell_to_label_counts.index[cell_type_iloc]
#     column_id = cell_to_label_counts.columns[column_id_iloc]
#     fake_nodes.append(
#         {
#             "cell_id": -i,
#             "column_id": column_id,
#             "cell_type": cell_type,
#         }
#     )
#     i += 1

# fake_nodes = pd.DataFrame(fake_nodes)
# fake_nodes.set_index("cell_id", inplace=True)
# fake_nodes
# nf.add_nodes(fake_nodes, inplace=True)

# %%

nf.nodes.sort_values(["column_id", "cell_type"], inplace=True)
nf.nodes

# %%
nf.query_edges("source != target", inplace=True)

# %%
nf.nodes["current_column_id"] = nf.nodes["column_id"].copy()
nf = nf.apply_node_features("current_column_id", inplace=False)

# %%
nf.edges
# %%
source_to_group_weights = (
    nf.edges.groupby(["source", "target_current_column_id"])["weight"]
    .sum()
    .to_frame()
    .reset_index()
)


group_to_target_weights = (
    nf.edges.groupby(
        [
            "target",
            "source_current_column_id",
        ]
    )["weight"]
    .sum()
    .to_frame()
    .reset_index()
)

# %%
group_to_target_weights.rename(
    {"target": "cell_id", "source_current_column_id": "current_column_id"},
    axis=1,
    inplace=True,
)

# %%

source_to_group_weights.rename(
    {"source": "cell_id", "target_current_column_id": "current_column_id"},
    axis=1,
    inplace=True,
)
# %%
group_weights = pd.concat([source_to_group_weights, group_to_target_weights], axis=0)

group_weights = group_weights.groupby(["cell_id", "current_column_id"])["weight"].sum()
# %%
group_weights

# %%
dummy_labels = np.concatenate([np.full(n_types, i) for i in range(1, n_columns + 1)])

group_df = pd.DataFrame(index=np.arange((n_ideal_cells)))
group_df["column_id"] = dummy_labels

# %%
uni_cell_types = np.unique(nf.nodes["cell_type"])

# %%
dummy_cell_types = np.concatenate([uni_cell_types for i in range(n_columns)])
# %%
group_df["cell_type"] = dummy_cell_types

# %%
group_df = group_df.set_index(["column_id", "cell_type"])
# %%
nf.nodes

# %%
cost_matrix = np.full((len(nf.nodes), n_ideal_cells), -1000)
cost_matrix = pd.DataFrame(cost_matrix, index=nf.nodes.index, columns=group_df.index)

# %%
group_weights
# %%

pbar = tqdm(total=len(group_weights))
for (cell_id, column_id), weight in group_weights.items():
    cost_matrix.loc[cell_id, (column_id, nf.nodes.loc[cell_id, "cell_type"])] = weight
    pbar.update(1)
pbar.close()

# %%

currtime = time.time()
row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
matched_node_index = cost_matrix.index[row_ind]
matched_group_index = cost_matrix.columns[col_ind]

matched_node_index = matched_node_index.to_frame().reset_index(drop=True)

matched_group_index = matched_group_index.to_frame().reset_index(drop=True)

# %%
matched_df = pd.concat([matched_node_index, matched_group_index], axis=1)
matched_df.set_index("cell_id", inplace=True)

# %%
missings = nf.nodes.query("~cell_id.isin(@matched_df.index) & cell_id > 0")[
    ["cell_type", "column_id"]
]

matched_df = pd.concat([matched_df, missings], axis=0)

# %%

matched_nf = NetworkFrame(nodes=matched_df, edges=nf.edges)

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
    matched_nf, "column_id"
)
print("New:")
print(new_n_within_group)
print(new_violations)
print()

# %%
