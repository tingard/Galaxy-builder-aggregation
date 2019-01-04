# Galaxy builder aggregation

This is a collection of scripts and jupyter notebooks used to perform data reduction and aggregation on the output classifications of Galaxy Builder

Current pipelines:

- [Spiral arm extraction](spiral-aggregation)
- [Disk, bulge, bar clustering](component-clustering)

Generic useful scripts can be found in `/lib`, which is symlinked to each of the working directories.

Don't even get me started on the list of Python dependancies....


## Requirements
- python 3 (ideally 3.6 or higher)
- `numpy`, `pandas`, `requests`, `scipy`, `matplotlib`, `scikit-learn`, `scikit-image`, `astropy`, `shapely`, `descartes`, `panoptes-aggregation` all available using `pip install ...`
- `gzbuilderspirals`, a custom package not on PyPi
