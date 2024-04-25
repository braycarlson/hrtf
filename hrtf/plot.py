from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from hrtf.dataset import Dataset


def channel(dataset: Dataset, row: pd.Series) -> None:
    figsize = (10, 8)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharey=True,
        figsize=figsize
    )

    fig.subplots_adjust(hspace=0.5)

    data = np.zeros([dataset.dimension, 2])

    data[:, 0] = dataset.impulse.get_values(
        indices = {
            'M': row.index,
            'R': 0,
            'E': 0
        }
    )

    data[:, 1] = dataset.impulse.get_values(
        indices = {
            'M': row.index,
            'R': 1,
            'E': 0
        }
    )

    ax1.plot(
        data[:, 0],
        label='Left',
        color='blue',
        linestyle='-'
    )

    ax1.grid(visible=True)

    ax2.plot(
        data[:, 1],
        label='Right',
        color='red',
        linestyle='-'
    )

    ax2.grid(visible=True)

    ax1.legend(loc='upper right')

    ax2.legend(loc='upper right')

    ax1.set_title(
        'Impulse Response for Left Channel',
        fontsize=14,
        pad=10
    )

    ax2.set_title(
        'Impulse Response for Right Channel',
        fontsize=14,
        pad=10
    )

    ax1.set_xlabel(
        'Time (ms)',
        fontsize=12,
        labelpad=10
    )

    ax1.set_ylabel(
        'Amplitude',
        fontsize=12,
        labelpad=10
    )

    ax2.set_xlabel(
        'Time (ms)',
        fontsize=12,
        labelpad=10
    )

    ax2.set_ylabel(
        'Amplitude',
        fontsize=12,
        labelpad=10
    )

    plt.show()
    plt.close()


def signal(dataset: Dataset, row: pd.Series) -> None:
    figsize = (10, 4)
    fig, ax = plt.subplots(figsize=figsize)

    data = np.zeros([dataset.dimension, 2])

    data[:, 0] = dataset.impulse.get_values(
        indices = {
            'M': row.index,
            'R': 0,
            'E': 0
        }
    )

    data[:, 1] = dataset.impulse.get_values(
        indices = {
            'M': row.index,
            'R': 1,
            'E': 0
        }
    )

    ax.plot(
        data[:, 0],
        label='Left',
        color='blue',
        linestyle='-'
    )

    ax.plot(
        data[:, 1],
        label='Right',
        color='red',
        linestyle='-'
    )

    ax.grid(visible=True)

    ax.legend(loc='upper right')

    ax.set_title(
        'Impulse Response',
        fontsize=14,
        pad=10
    )

    ax.set_xlabel(
        'Time (ms)',
        fontsize=12,
        labelpad=10
    )

    ax.set_ylabel(
        'Amplitude',
        fontsize=12,
        labelpad=10
    )

    plt.show()
    plt.close()
