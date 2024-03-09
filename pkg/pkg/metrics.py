def cell_to_column_crosstab(nodes, label_feature="column_id", dropna=False):
    out = (
        nodes.pivot_table(
            index="cell_type", columns=label_feature, aggfunc="size", dropna=dropna
        )
        .fillna(0)
        .astype(int)
    )
    return out


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
