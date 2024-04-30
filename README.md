## A Head-related Transfer Function (HRTF) Demonstration

A demonstration of the Head-related Transfer Function (HRTF) and convolving audio by [Angela De Sousa Costa](https://github.com/angeladesousacosta), [Brayden Carlson](https://github.com/braycarlson) and [Tyrell Martens](https://github.com/Hotrod1220). You can select a point from the scatter plot, which will attempt to match the closest impulse response by its azimuth and elevation, and convolve the audio using that matching impulse response. This will simulate the experience of directional audio. Each time a point is selected, it generates a figure that illustrates the original audio, convolved audio, and impulse response.

![A screenshot of the demonstration](asset/demonstration.png?raw=true "Demonstration")

## Prerequisites

* [pyenv](https://github.com/pyenv/pyenv) or [Python 3.11.2](https://www.python.org/downloads/)


## Setup

### pyenv

```
pyenv install 3.11.2
```

```
pyenv local 3.11.2
```

### Virtual Environment

```
python -m venv venv
```

#### Windows

```
"venv/Scripts/activate"
```

#### Unix

```
source venv/bin/activate
```

### Packages

```
pip install -U -r requirements.txt
```
