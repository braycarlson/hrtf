from __future__ import annotations

from math import sqrt, atan2, degrees


def calculate_azimuth(x: tuple[float, ...]) -> float:
    x1, y1, _ = x

    azimuth = degrees(
        atan2(
            y1,
            x1
        )
    )

    if azimuth < 0:
        azimuth = azimuth + 360

    return azimuth


def calculate_elevation(x: tuple[float, ...]) -> float:
    x1, y1, z1 = x

    return degrees(
        atan2(
            z1,
            sqrt(x1**2 + y1**2)
        )
    )


def calculate_radius(x: tuple[float, ...]) -> float:
    x1, y1, z1 = x

    return sqrt(
        x1**2 +
        y1**2 +
        z1**2
    )
