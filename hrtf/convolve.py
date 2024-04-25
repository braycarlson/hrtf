from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from hrtf.constant import ANIMATION, DATA
from hrtf.dataset import Dataset
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from scipy import signal
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.lines import Line2D
    from pathlib import Path


class Visualizer:
    def __init__(
        self,
        animation: Path | None = None,
        dataset: Path | None = None,
        original: Path | None = None
    ):
        self.animation = animation
        self.dataset = dataset
        self.original = original

    @staticmethod
    def convolve(
        audio: npt.NDArray,
        impulse: npt.NDArray,
        dimension: int
    ) -> npt.NDArray:
        empty = np.zeros(
            [dimension, 2]
        )

        mono = np.mean(audio, axis=1)

        left = signal.fftconvolve(
            mono,
            impulse[:, 0]
        )

        right = signal.fftconvolve(
            mono,
            impulse[:, 1]
        )

        # Zero pad audio, if needed
        if len(empty) < len(left):
            difference = len(left) - len(empty)

            empty = np.append(
                empty,
                np.zeros([difference, 2]),
                0
            )

        empty = np.vstack([left, right]).transpose()
        return empty / np.max(np.abs(empty))

    def compute(
        self,
        index: int,
        a: int,
        e: int,
    ) -> None:
        original, rate = sf.read(self.original)

        dataset = Dataset.load(self.dataset)
        dataset.rate = rate

        audio = original.copy()

        impulse = dataset.get_data(index)

        convolution = self.convolve(
            audio,
            impulse,
            dataset.dimension
        )

        duration = len(convolution) / dataset.rate

        audio = audio.flatten()
        convolution = convolution.flatten()
        impulse = impulse.flatten()

        maximum = max(
            len(audio),
            len(convolution),
            len(impulse)
        )

        length = len(audio)

        audio = np.pad(
            audio,
            (0, maximum - length)
        )

        length = len(convolution)

        convolution = np.pad(
            convolution,
            (0, maximum - length)
        )

        length = len(impulse)

        impulse = np.pad(
            impulse,
            (0, maximum - length)
        )

        downsample = 1250

        time = np.linspace(
            0,
            duration,
            maximum
        )[::downsample]

        audio = audio[::downsample]
        convolution = convolution[::downsample]
        impulse = impulse[::downsample]

        audio = audio[:len(time)]
        convolution = convolution[:len(time)]
        impulse = impulse[:len(time)]

        ylim = max(*audio, *convolution, *impulse)

        fig, ax = plt.subplots()
        ax.patch.set_facecolor('#dddddd')

        ax.set_xlim(0, duration)
        ax.set_ylim(-ylim - 0.05, ylim + 0.05)

        lines = {
            'audio': None,
            'convolved': None,
            'hrtf': None
        }

        keys = lines.keys()
        values = lines.values()

        iterable = [audio, convolution, impulse]

        for label, _ in zip(keys, iterable):
            empty = []

            lines[label], = ax.plot(
                empty,
                empty,
                label=label,
                lw=1,
            )

        def init_func() -> list[Line2D, Line2D, Line2D]:
            for line in lines.values():
                empty = []

                line.set_data(
                    empty,
                    empty
                )

            return lines.values()

        def update(frame: int) -> list[Line2D, Line2D, Line2D]:
            for _, y, line in zip(keys, iterable, values):
                line.set_data(
                    time[:frame],
                    y[:frame]
                )

                if frame > dataset.rate:
                    ax.set_xlim(
                        time[frame - dataset.rate],
                        time[frame]
                    )

                    y_range = y[frame - dataset.rate:frame]

                    if len(y_range) > 0:
                        ax.set_ylim(
                            y_range.min(),
                            y_range.max()
                        )

            return lines.values()

        length = len(time)
        frames = range(length)

        interval = int(
            (duration * 1000.0) / len(time)
        )

        labelpad, pad = 10, 10

        title = f"Convolution between Applause and IR: Azimuth: {a}° and Elevation: {e}°"

        ax.set_title(
            title,
            fontsize=10,
            pad=pad
        )

        ax.set_xlabel(
            'Time (s)',
            fontsize=12,
            labelpad=labelpad
        )

        ax.set_ylabel(
            'Amplitude',
            fontsize=12,
            labelpad=labelpad
        )

        legend = ax.legend()
        legend.get_frame().set_facecolor('#dddddd')

        plt.tight_layout()

        animation = FuncAnimation(
            fig,
            update,
            blit=True,
            frames=frames,
            init_func=init_func,
            interval=interval,
        )

        filename = f"{index}.mp4"
        path = ANIMATION.joinpath(filename)

        animation.save(path, writer='ffmpeg')
        plt.close()

    def run(self) -> None:
        path = DATA.joinpath('source.csv')
        dataframe = pd.read_csv(path)

        azimuths = dataframe.azimuth.tolist()
        elevations = dataframe.elevation.tolist()

        total = len(dataframe)
        iterable = zip(azimuths, elevations)

        Parallel(n_jobs=6)(
            delayed(self.compute)(index, azimuth, elevation)
            for index, (azimuth, elevation) in tqdm(enumerate(iterable), total=total)
        )


def main() -> None:
    original = DATA.joinpath('sample/applause.wav')
    dataset = DATA.joinpath('rir/H3_44K_16bit_256tap_FIR_SOFA.sofa')

    visualizer = Visualizer(
        dataset=dataset,
        original=original
    )

    visualizer.run()


if __name__ == '__main__':
    main()
