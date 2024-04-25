from __future__ import annotations

import librosa
import numpy as np
import numpy.typing as npt

from sofa import Database
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from sofa.access.variables import Variable
    from typing_extensions import Self


class Dataset:
    def __init__(
        self,
        path: Path | None = None,
        rate: float | None = None
    ):
        self.database = None
        self.mapping = None
        self.path = path
        self.rate = rate

    @property
    def dimension(self) -> int:
        return self.database.Dimensions.N

    @property
    def impulse(self) -> Variable:
        return self.database.Data.IR

    @property
    def listener(self) -> npt.NDArray:
        return self.database.Listener.Position.get_values(system='spherical')

    @property
    def receiver(self) -> npt.NDArray:
        return self.database.Receiver.Position.get_values(system='spherical')

    @property
    def source(self) -> npt.NDArray:
        return self.database.Source.Position.get_values(system='spherical')

    @property
    def target(self) -> float:
        target, *_ = self.database.Data.SamplingRate.get_values()
        return target

    @classmethod
    def load(
        cls: type[Self],
        path: Path | None = None,
        rate: float | None = None
    ) -> Self:
        dataset = cls(path, rate)

        path = str(path)
        dataset.database = Database.open(path)

        dataset.mapping = {}

        for i, (azimuth, elevation, radius) in enumerate(dataset.source):
            dataset.mapping[(azimuth, elevation, radius)] = i

        return dataset

    def get_angle(self, azimuth: int, elevation: int) -> int:
        azimuth_difference = np.abs(self.source[:, 0] - azimuth)
        elevation_difference = np.abs(self.source[:, 1] - elevation)

        difference = azimuth_difference + elevation_difference

        return np.argmin(difference)

    def get_data(self, index: int) -> npt.NDArray:
        data = np.zeros([self.dimension, 2])

        data[:, 0] = self.impulse.get_values(
            indices = {
                'M': index,
                'R': 0,
                'E': 0
            }
        )

        data[:, 1] = self.impulse.get_values(
            indices = {
                'M': index,
                'R': 1,
                'E': 0
            }
        )

        if self.rate == self.target:
            return data

        return librosa.core.resample(
            data,
            orig_sr=self.target,
            target_sr=self.rate
        )

    def get_index(self, azimuth: float, elevation: float) -> int:
        return self.mapping.get(
            (azimuth, elevation),
            None
        )
