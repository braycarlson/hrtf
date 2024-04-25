from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from IPython.display import display
from ipywidgets import (
    HBox,
    HTML,
    Layout,
    VBox,
)

from hrtf.spatial import (
    calculate_azimuth,
    calculate_elevation,
    calculate_radius
)
from pathlib import Path
from plotly import graph_objs as go
from scipy import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from dataset import Dataset
    from typing_extensions import Self


class Builder:
    def __init__(
        self,
        audio: npt.NDArray = None,
        dataframe: pd.DataFrame = None,
        dataset: Dataset = None,
        mesh: go.Scatter3d = None,
        source: go.Scatter3d = None,
        transfer: str | None = None
    ):
        self.audio = audio
        self.convolution = None
        self.dataframe = dataframe
        self.dataset = dataset
        self.hrtf = None
        self.index = 0
        self.mesh = mesh
        self.source = source
        self.transfer = transfer
        self.video = HTML(value='')

        self.plot = Plot()

        self.height = 900
        self.width = 1600

    def convolve(self) -> Self:
        audio = np.zeros(
            [self.dataset.dimension, 2]
        )

        azimuth = self.dataframe.loc[0, 'azimuth'].squeeze()
        elevation = self.dataframe.loc[0, 'elevation'].squeeze()

        self.index = self.dataset.get_angle(azimuth, elevation)
        self.hrtf = self.dataset.get_data(self.index)

        mono = np.mean(self.audio, axis=1)

        left = signal.fftconvolve(
            mono,
            self.hrtf[:, 0]
        )

        right = signal.fftconvolve(
            mono,
            self.hrtf[:, 1]
        )

        # Zero pad audio if needed
        if len(audio) < len(left):
            difference = len(left) - len(audio)

            audio = np.append(
                audio,
                np.zeros([difference, 2]),
                0
            )

        audio = np.vstack([left, right]).transpose()
        return audio / np.max(np.abs(audio))

    def css(self) -> Self:
        stylesheet = Path.cwd().joinpath('stylesheet/stylesheet.css')

        with open(stylesheet, 'r') as handle:
            file = handle.read()

        css = f"""
            <style>
                {file}
            </style>
        """

        css = HTML(css)
        display(css)

        return self

    def get(self) -> VBox:
        html = self.plot.component.get('html')
        widget = self.plot.component.get('widget')

        left = VBox(
            [html, self.video],
            layout=Layout(
                display='flex',
                justify_content='center',
                align_items='center'
            )
        )

        left.add_class('left')

        right = VBox(
            [widget],
            layout=Layout(
                display='flex',
                justify_content='center',
                align_items='center',
            )
        )

        right.add_class('right')

        element = [left, right]

        return VBox(
            [HBox(element)],
            layout=Layout(
                display='flex',
                align_items='center',
                height='100%',
                overflow='hidden'
            )
        )

    def html(self) -> Self:
        value = (
            self.dataframe
            .transpose()
            .to_html(
                classes='description',
                index=True,
                justify='center'
            )
        )

        html = HTML(value=value)
        self.plot.component['html'] = html

        return self

    def layout(self) -> Self:
        scene = self.plot.component.get('scene')

        layout = go.Layout(
            autosize=False,
            width=1600,
            height=900,
            margin=go.layout.Margin(
                l=30,
                r=30,
                t=30,
                b=30
            ),
            title='',
            font={'size': 11},
            scene=scene,
            template='ggplot2'
        )

        self.plot.component['layout'] = layout

        return self

    def limit(self) -> Self:
        x = [
            min(
                np.min(self.mesh.x),
                np.min(self.source.x)
            ),
            max(
                np.max(self.mesh.x),
                np.max(self.source.x)
            )
        ]

        y = [
            min(
                np.min(self.mesh.y),
                np.min(self.source.y)
            ),
            max(
                np.max(self.mesh.y),
                np.max(self.source.y)
            )
        ]

        z = [
            min(
                np.min(self.mesh.z),
                np.min(self.source.z)
            ),
            max(
                np.max(self.mesh.z),
                np.max(self.source.z)
            )
        ]

        self.plot.component['limit'] = (x, y, z)

        return self

    def event(self) -> Self:
        widget = self.plot.component.get('widget')

        for trace in widget.data:
            trace.on_click(self.on_click)

        return self

    def on_click(
        self,
        trace: go.Scatter3d,
        points: go.callbacks.Point,
        _ : None
    ) -> None:
        if not points.point_inds:
            return

        self.video.value = ''

        index = points.point_inds[0]

        coordinates = (
            trace.x[index],
            trace.y[index],
            trace.z[index]
        )

        x, y, z = coordinates

        self.dataframe.loc[0, 'x'] = round(x, 2)
        self.dataframe.loc[0, 'y'] = round(y, 2)
        self.dataframe.loc[0, 'z'] = round(z, 2)

        radius = calculate_radius(coordinates)
        azimuth = calculate_azimuth(coordinates)
        elevation = calculate_elevation(coordinates)

        self.dataframe.loc[0, 'radius'] = round(radius, 2)
        self.dataframe.loc[0, 'azimuth'] = round(azimuth, 2)
        self.dataframe.loc[0, 'elevation'] = round(elevation, 2)

        html = self.plot.component.get('html')

        html.value = (
            self.dataframe
            .transpose()
            .to_html(
                classes='information',
                index=True,
                justify='center'
            )
        )

        self.play()

    def play(self) -> None:
        convolution = self.convolve()
        sd.play(convolution, self.dataset.rate)

        self.video.value = self.transfer.load(self.index)

        # self.transfer.convolution = convolution
        # self.transfer.hrtf = self.hrtf

        # self.video.value = self.transfer.create()

        return self

    def scene(self) -> Self:
        limit = self.plot.component.get('limit')

        x_range, y_range, z_range = limit

        scene = go.layout.Scene(
            xaxis={'range': x_range},
            yaxis={'range': y_range},
            zaxis={'range': z_range},
            camera = {
                'center': {'x': 0, 'y': 0, 'z': 0},
                'eye': {'x': 1.25, 'y': 2.5, 'z': 1.25},
                'up': {'x': 0, 'y': 0, 'z': 1}
            }
        )

        self.plot.component['scene'] = scene

        return self

    def widget(self) -> Self:
        layout = self.plot.component.get('layout')

        data = [self.source, self.mesh]

        widget = go.FigureWidget(
            data=data,
            layout=layout
        )

        self.plot.component['widget'] = widget

        return self


class Plot:
    def __init__(self):
        self.component = {}


class Interactive:
    def __init__(self, builder: Builder | None = None):
        self.builder = builder

    @property
    def audio(self) -> npt.NDArray:
        return self.builder.audio

    @audio.setter
    def audio(self, audio: npt.NDArray) -> None:
        self.builder.audio = audio

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.builder.dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        self.builder.dataframe = dataframe

    @property
    def dataset(self) -> Dataset:
        return self.builder.dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        self.builder.dataset = dataset

    @property
    def mesh(self) -> go.Scatter3d:
        return self.builder.mesh

    @mesh.setter
    def mesh(self, mesh: go.Scatter3d) -> None:
        self.builder.mesh = mesh

    @property
    def source(self) -> go.Scatter3d:
        return self.builder.source

    @source.setter
    def source(self, source: go.Scatter3d) -> None:
        self.builder.source = source

    @property
    def transfer(self) -> str:
        return self.builder.transfer

    @transfer.setter
    def transfer(self, transfer: str) -> None:
        self.builder.transfer = transfer

    def create(self) -> VBox:
        return (
            self.builder
            .limit()
            .scene()
            .layout()
            .widget()
            .html()
            .css()
            .event()
            .get()
        )
