# Basic Image Segmentation

This is a codebase for changing background of a given image to another.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

### Prerequisites

- Python 3.x
- [pip](https://pip.pypa.io/en/stable/installation/)
- [miniconda](https://docs.anaconda.com/miniconda/)

### Clone the Repository

```sh
git clone https://github.com/Yash-Vardhan-Sharma/image-segmentation.git
cd image-segmentation
```
- Create a conda environment for the project

```sh
conda create --name <name-of-proj> python=3.x
conda activate <name-of-proj>
```

## Usage

```sh
pip install -r requirements.txt
python main.py -i <path-of-image-file> -b <path-of-background-file>
```

This will save the modified file in the `saved_image` directory.
`images` and `backgrounds` directory containg sample images to modify.

Example use:

```
python main.py -i images/sample_image.png -b backgrounds/background1.png
```





