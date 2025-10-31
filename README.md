# panelbeater

<a href="https://pypi.org/project/panelbeater/">
    <img alt="PyPi" src="https://img.shields.io/pypi/v/panelbeater">
</a>

A CLI for finding mispriced options.

## Dependencies :globe_with_meridians:

Python 3.11.6:

- [yfinance](https://ranaroussi.github.io/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/)
- [numpy](https://numpy.org/)
- [feature-engine](https://feature-engine.trainindata.com/en/latest/)
- [requests-cache](https://requests-cache.readthedocs.io/en/stable/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [wavetrainer](https://github.com/8W9aG/wavetrainer/)
- [tqdm](https://tqdm.github.io/)

## Raison D'être :thought_balloon:

`panelbeater` trains models at t+X iteratively to come up with the calibrated expected distribution of an asset price in the future. It then finds the current prices of options for an asset, and determines whether it should be bought and for how much.

## Architecture :triangular_ruler:

`panelbeater` goes through the following steps:
1. Downloads the historical data.
2. Performs feature engineering on the data.
3. Trains the required models to operate on the data panel.
4. Downloads the current data.
5. Runs inference on t+X for the latest options to find the probability distribution on the asset prices to their expiry dates.
6. Finds any mispriced options and size the position accordingly.

## Installation :inbox_tray:

This is a python package hosted on pypi, so to install simply run the following command:

`pip install panelbeater`

or install using this local repository:

`python setup.py install --old-and-unmanageable`

## Usage example :eyes:

You can run `panelbeater` as a CLI like so:

```shell
panelbeater
```

This performs a full train, inference and attempts to find mispriced options.

## License :memo:

The project is available under the [MIT License](LICENSE).
