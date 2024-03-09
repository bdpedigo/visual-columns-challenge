import numpy as np
import pandas as pd
from networkframe import NetworkFrame

from .constants import N_COLUMNS, N_TYPES
from .metrics import cell_to_column_crosstab


def add_fake_nodes(nf, label_feature="column_id"):
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


def create_target_nodes(nf):
    dummy_labels = np.concatenate(
        [np.full(N_TYPES, i) for i in range(1, N_COLUMNS + 1)]
    )
    uni_cell_types = np.unique(nf.nodes["cell_type"])
    dummy_cell_types = np.concatenate([uni_cell_types for _ in range(N_COLUMNS)])
    target_nodes = pd.DataFrame(
        {
            "cell_type": dummy_cell_types,
            "column_id": dummy_labels,
            "node_type": "target",
        }
    )
    target_nodes

    cell_type_counts = nf.nodes["cell_type"].value_counts().sort_index()
    cell_type_extras = cell_type_counts - N_COLUMNS

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

    return target_nodes


def create_matching_target(target_nodes):
    target_labels = target_nodes["column_id"].fillna(-1).values
    mask = (target_labels[:, None] == target_labels[None, :]).astype(float)
    not_fake = (target_nodes["column_id"].notna()).values
    not_fake_mask = not_fake[:, None] & not_fake[None, :]
    B = (mask * not_fake_mask).astype(float)

    return B


def create_matched_networkframe(nf, result, target_nodes):
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


def correct_violations(nf, label_feature):
    edges = nf.apply_node_features(label_feature).edges
    within_group_edges = edges.query(
        f"source_{label_feature} == target_{label_feature}"
    )

    nodes = nf.nodes
    # don't count the dummy nodes here
    nodes = nodes[nodes.index > 0]
    cell_to_label_counts = (
        nodes.pivot_table(index="cell_type", columns=label_feature, aggfunc="size")
        .fillna(0)
        .astype(int)
    )

    violations_mask = cell_to_label_counts > 1
    violation_groups = np.where(violations_mask)

    corrected_nodes = nf.nodes.copy()

    for group_i in range(len(violation_groups[0])):
        cell_type_iloc = violation_groups[0][group_i]
        column_id_iloc = violation_groups[1][group_i]
        cell_type = cell_to_label_counts.index[cell_type_iloc]  # noqa: F841
        column_id = cell_to_label_counts.columns[column_id_iloc]  # noqa: F841
        cell_type
        column_id
        
        conflicting_nodes = nodes.query(
            "cell_type == @cell_type & predicted_column_id == @column_id"
        )

        # what would happen if we unassigned these nodes?
        # test them one at a time, find the unassignment which would
        # lose the least amount of within-group edges
        best_score = 0
        best_node = None
        for node in conflicting_nodes.index:
            others = conflicting_nodes.index[conflicting_nodes.index != node]
            # subset the data to only count edges where this node is in that group
            subset_edges = within_group_edges.query(
                "~source.isin(@others) & ~target.isin(@others)"
            )
            subset_score = subset_edges["weight"].sum()
            if subset_score > best_score:
                best_score = subset_score
                best_node = node

        others = conflicting_nodes.index[conflicting_nodes.index != best_node]
        corrected_nodes.loc[others, "predicted_column_id"] = np.nan
        corrected_nodes.loc[others, "predicted_node_type"] = "extra"

    corrected_nf = NetworkFrame(
        nodes=corrected_nodes,
        edges=nf.edges,
    )
    corrected_nf.query_nodes(
        "node_type == 'real' & predicted_node_type == 'target'", inplace=True
    )
    return corrected_nf
