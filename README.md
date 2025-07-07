<!-- [![PyPI version fury.io](https://badge.fury.io/py/geoxyz-scraper.svg)](https://pypi.python.org/pypi/geoxyz-scraper/)
[![Documentation Status](https://readthedocs.org/projects/geoxyz-scraper/badge/?version=latest)](https://geoxyz-scraper.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/geoxyz-scraper/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/geoxyz-scraper/blob/main/.github/workflows/tests.yml)
[![codecov](https://codecov.io/gh/martibosch/geoxyz-scraper/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/geoxyz-scraper) -->

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/geoxyz-scraper/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/geoxyz-scraper/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/geoxyz-scraper.svg)](https://github.com/martibosch/geoxyz-scraper/blob/main/LICENSE)

# GeoXYZ Scraper

Python package to scrape geospatial XYZ tiles.

## Installation

The best way to install "geoxyz-scraper" is first installing GDAL using conda/mamba:

```bash
conda install -c conda-forge gdal
```

and then, within the same environment, use pip/uv to install "geoxyz-scraper"

```bash
pip install https://github.com/martibosch/geoxyz-scraper/archive/main.zip
```

## Usage

Run the tool from the command line as in:

```bash
geoxyz-scraper <path-to-aoi.gpkg> <path-to-config.yml> <dst-dir>
```

where:

- `<path-to-aoi.gpkg>` is a geopackage file containing the area of interest (AOI) polygon.
- `<path-to-config.yml>` is a YAML file containing the configuration for the XYZ tile source.
- `<dst-dir>` is the destination directory where the scraped tiles will be saved.

additional options can be seen by running:

```bash
geoxyz-scraper --help
```

## Acknowledgements

- This package was created with the [martibosch/cookiecutter-geopy-package](https://github.com/martibosch/cookiecutter-geopy-package) project template.
