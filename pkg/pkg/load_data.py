import numpy as np
import pandas as pd
from networkframe import NetworkFrame

from .constants import N_COLUMNS
from .paths import DATA_PATH


def load_networkframe(sample=False, seed=8888):
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
        columns={
            "from_cell_id": "source",
            "to_cell_id": "target",
            "synapses": "weight",
        },
        inplace=True,
    )

    nf = NetworkFrame(
        nodes=columns_df,
        edges=connections_df,
    )

    nf.nodes["node_type"] = "real"

    if sample:
        np.random.seed(seed)

        n_select_columns = 10
        select_cols = np.random.choice(
            np.arange(1, N_COLUMNS + 1), size=n_select_columns, replace=False
        )
        select_cols
        unassigned_nodes = (
            nf.nodes.query("column_id.isna()")
            .index.to_series()
            .sample(frac=n_select_columns / N_COLUMNS)
        )
        unassigned_nodes
        nf = nf.query_nodes(
            "column_id.isin(@select_cols) | index.isin(@unassigned_nodes)",
            local_dict=locals(),
        ).copy()

    return nf
