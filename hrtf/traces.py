from __future__ import annotations

import base64
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import trimesh

from matplotlib.animation import FuncAnimation
from pathlib import Path
from plotly import graph_objs as go
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrtf.dataset import Dataset
    from matplotlib.lines import Line2D


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class TransferFunctionTrace:
    def __init__(self, audio: npt.NDArray = None):
        self.audio = audio
        self.convolution = None
        self.hrtf = None
        self.path = Path.cwd().joinpath('animation')
        self.rate = None

    def create(self) -> None:
        duration = len(self.convolution) / self.rate

        audio = self.audio.flatten()
        convolution = self.convolution.flatten()
        hrtf = self.hrtf.flatten()

        maximum = max(
            len(audio),
            len(convolution),
            len(hrtf)
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

        length = len(hrtf)

        hrtf = np.pad(
            hrtf,
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
        hrtf = hrtf[::downsample]

        audio = audio[:len(time)]
        convolution = convolution[:len(time)]
        hrtf = hrtf[:len(time)]

        ylim = max(*audio, *convolution, *hrtf)

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

        iterable = [audio, convolution, hrtf]

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

                if frame > self.rate:
                    ax.set_xlim(
                        time[frame - self.rate],
                        time[frame]
                    )

                    y_range = y[frame - self.rate:frame]

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

        ax.set_title(
            'Convolutional Waveform',
            fontsize=14,
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

        return animation.to_html5_video()

    def load(self, index: int) -> None:
            video = """<video width="320" height="240" controls autoplay>
                <source src="data:video/mp4;base64,{0}" type="video/mp4">
            </video>"""

            filename = f"{index}.mp4"
            path = self.path.joinpath(filename)

            with open(path, 'rb') as handle:
                file = handle.read()

                encoded = base64.b64encode(file).decode('ascii')
                return video.format(encoded)


class MeshTrace:
    def __init__(self, path: Path | None = None):
        self.mesh = trimesh.load_mesh(path)
        self.path = path

    def create(self) -> go.Scatter3d:
        angle_z = np.pi / 2

        transform_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0, 0],
            [np.sin(angle_z), np.cos(angle_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.mesh.apply_transform(transform_z)

        self.mesh.apply_scale(0.005)

        center = self.mesh.vertices.mean(axis=0)

        identity = np.eye(4)
        identity[:3, 3] = -center

        self.mesh.apply_transform(identity)

        self.x, self.y, self.z = self.mesh.vertices.T
        self.i, self.j, self.k = self.mesh.faces.T

        return go.Mesh3d(
            x=self.x,
            y=self.y,
            z=self.z,
            i=self.i,
            j=self.j,
            k=self.k,
            color='darkgrey',
            showscale=False,
            hoverinfo='none',
            opacity=1.0,
            lighting={
                'ambient': 0,
                'diffuse': 1,
                'fresnel': 1,
                'roughness': 0.05,
                'specular': 0.5
            },
            lightposition={
                'x': 100,
                'y': 100,
                'z': 100
            },
        )


class ListenerScatterTrace:
    def __init__(self, dataset: Dataset, radius: float = 1.2):
        self.dataset = dataset
        self.radius = radius

        azimuth, elevation = (
            np.radians(dataset.listener[:, 0]),
            np.radians(dataset.listener[:, 1])
        )

        self.x = self.radius * np.cos(elevation) * np.cos(azimuth)
        self.y = self.radius * np.cos(elevation) * np.sin(azimuth)
        self.z = self.radius * np.sin(elevation)

        self.n = len(self.x)

    def create(self) -> go.Scatter3d:
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='markers',
            marker={
                'color': 'red',
                'opacity': 0.50,
                'size': 25
            },
            hoverinfo='none'
        )


class ReceiverScatterTrace:
    def __init__(self, dataset: Dataset, radius: float = 1.2):
        self.dataset = dataset
        self.radius = radius

        azimuth, elevation = (
            np.radians(dataset.receiver[:, 0]),
            np.radians(dataset.receiver[:, 1])
        )

        self.x = self.radius * np.cos(elevation) * np.cos(azimuth)
        self.y = self.radius * np.cos(elevation) * np.sin(azimuth)
        self.z = self.radius * np.sin(elevation)

        self.n = len(self.x)

    def create(self) -> go.Scatter3d:
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='markers',
            marker={
                'color': 'red',
                'opacity': 0.50,
                'size': 50
            },
            hoverinfo='none'
        )


class SourceScatterTrace:
    def __init__(self, dataset: Dataset, radius: float = 1.2):
        self.dataset = dataset
        self.radius = radius

        azimuth, elevation = (
            np.radians(dataset.source[:, 0]),
            np.radians(dataset.source[:, 1])
        )

        self.x = self.radius * np.cos(elevation) * np.cos(azimuth)
        self.y = self.radius * np.cos(elevation) * np.sin(azimuth)
        self.z = self.radius * np.sin(elevation)

        self.n = len(self.x)
        self.color = elevation

    def create(self) -> go.Scatter3d:
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='markers',
            marker={
                'color': self.color,
                'colorscale': 'Portland',
                'opacity': 0.50,
                'size': 5,
                'colorbar': {
                    'title': 'Elevation'
                }
            },
            hoverinfo='none'
        )


class ArbitraryScatterTrace:
    def __init__(self, n: int = 500, radius: float = 1.2):
        self.n = n
        self.radius = radius

        self.azimuth = np.radians(
            np.linspace(
                0,
                360,
                int(
                    np.round(
                        np.sqrt(self.n)
                    )
                )
            )
        )

        self.elevation = np.radians(
            np.linspace(
                -30,
                80,
                int(
                    np.round(
                        np.sqrt(self.n)
                    )
                )
            )
        )

    def create(self) -> go.Scatter3d:
        self.mesh = azimuth, elevation = np.meshgrid(
            self.azimuth,
            self.elevation
        )

        x = self.radius * np.cos(elevation) * np.cos(azimuth)
        y = self.radius * np.cos(elevation) * np.sin(azimuth)
        z = self.radius * np.sin(elevation)

        self.x, self.y, self.z = (
            x.ravel(),
            y.ravel(),
            z.ravel()
        )

        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='markers',
            marker={
                'color': 'black',
                'opacity': 0.50,
                'size': 5
            },
            hoverinfo='none'
        )
