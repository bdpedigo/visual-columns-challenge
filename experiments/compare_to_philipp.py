# %%
import pandas as pd
from pkg import DATA_PATH

annotations_file = DATA_PATH / "Supplemental_file1_neuron_annotations.tsv"

annotations = pd.read_csv(annotations_file, sep="\t", low_memory=False)
annotations.set_index("root_id", inplace=True)

# %%
from pkg import load_networkframe

nf = load_networkframe()

# %%

nf.nodes = nf.nodes.join(annotations, how="inner", rsuffix="_schlegel")

# %%
nf.nodes

# %%
nf.nodes.columns

# %%
nf.nodes["cell_type_schlegel"]

# %%
has_cell_type_nodes = nf.nodes.query("cell_type_schlegel.notna()")

#%%
len(has_cell_type_nodes) / len(nf.nodes)

# %%
# has_cell_type_nodes.query("cell_type == cell_type_schlegel").sum() / len(
#     has_cell_type_nodes
# )
(has_cell_type_nodes["cell_type"] == has_cell_type_nodes["cell_type_schlegel"]).mean()

# %%