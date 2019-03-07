# Galaxy builder aggregation

This is a collection of scripts and jupyter notebooks used to perform data reduction and aggregation on the output classifications of Galaxy Builder

Current pipelines:

- [Spiral arm extraction](spiral-aggregation)
- [Disk, bulge, bar clustering](component-clustering)
- ["Best Individual Classification"](model-scoring)

Generic useful scripts can be found in `/lib`, which is symlinked to each of the working directories.

Classification and subject exports can be found in `/classifications`

Apologies for the random scripts. I plan on cleaning up this repository eventually but need to allocate time to do it.

## Requirements
- python 3 (ideally 3.6 or higher)
- `numpy`, `pandas`, `requests`, `scipy`, `matplotlib`, `scikit-learn`, `scikit-image`, `astropy`, `shapely`, `descartes`, `panoptes-aggregation` all available using `pip install ...`
- `gzbuilderspirals`, a custom package not on PyPi ([available on github](https://github.com/tingard/gzbuilderspirals))
