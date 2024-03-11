import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import csr_array


def plot_matched_matrices(A, B, S, result=None):
    if isinstance(A, csr_array):
        A = A.toarray()
    if isinstance(B, csr_array):
        B = B.toarray()
    if isinstance(S, csr_array):
        S = S.toarray()
    if result is None:
        indices_A = np.arange(A.shape[0])
        indices_B = np.arange(B.shape[0])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ax = axs[0]
    sns.heatmap(
        A[indices_A][:, indices_A] > 0,
        ax=ax,
        cbar=False,
        square=True,
        cmap="RdBu_r",
        center=0,
    )
    ax.set_title("A")

    ax = axs[1]
    sns.heatmap(
        B[indices_B][:, indices_B],
        ax=ax,
        cbar=False,
        square=True,
        cmap="RdBu_r",
        center=0,
    )
    ax.set_title("B")

    ax = axs[2]
    sns.heatmap(
        S[indices_A][:, indices_B],
        ax=ax,
        cbar=False,
        square=True,
        cmap="RdBu_r",
        center=0,
    )
    ax.set_title("S")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs
