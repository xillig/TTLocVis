
# TTLocVis

A Twitter Topic Location Visualization Python package

## Summary   

The package TTLocVis provides a broad range of methods to generate, clean, analyze and visualize the contents of Twitter
data. TTLocVis enables the user to work with geo-spatial Twitter data and to generate topic distributions from Latent 
Dirichlet Allocation (LDA) Topic Models for geo-coded Tweets. As such, TTLocVis is an innovative 
tool to work with geo-coded text on a high geo-spatial resolution to analyse the public discourse on various topics in 
space and time. The package can be used for a broad range of applications for scientific research to gain insights into 
topics discussed on Twitter. 

In general, Topic Models are generative probabilistic models, that provide an insight into hidden information 
in large text corpora by estimating the underlying topics of the texts in an unsupervised manner.

Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world.
Subsequently, the inherently noisy Twitter data can be cleaned, transformed and exported. 
In particular, TTLocVis enables the user to apply LDA Topic Models on extremely sparse Twitter data by preparing 
the Tweets for LDA analysis by the pooling Tweets by hashtags.

TTLocVis provides options for automatized Topic Model parameter optimization. Furthermore, a distribution over 
topics is generated for each document. The distribution of topics over documents can be visualized with various 
plotting methods. The average prevalence of topics in the documents at each day can 
be plotted as a time series, in order to visualize, how topics develop over time.
 
Above this, the spatial distribution of Tweets can be plotted on a world map, which automatically chooses an appropriate
part of the world, in order to visualise the chosen sample of Tweets. As part of the mapping process, each Tweet is 
classified by its most prevalent topic and colour coded.

## How to cite 

TBA

# Installation

__Attention:__ Event though TTLocVis should run on Python 3.7 and 3.8, it was not fully tested under these conditions.
We do recommend to install a new (conda) environment with Python 3.6. 

The package can be installed via *pip*:
```commandline
python pip install TTLocVis
```

### Windows

After successful installation, the user must download the [*basemap* package] and install it manually via *pip*:
```commandline
python -m pip install [path-to-the-downloaded-file/your-basemap-wheel]
```
__Note:__ Do not copy the name of your *basemap wheel* from the above mentioned website into your python console! Write
it out manually!
The *cpXX* in the filenames refer to the python version you will use. An example for Python 3.6. would be the file 
*basemap-1.2.1-cp36-cp36m-win_amd64.whl* Remember, TTLocVis is developed to run only on Python 3.6, 3.7 and 3.8.

[*basemap* package]: https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap

### Linux and iOS

Download [basemap package version 1.2.1] and install it accordingly.

[basemap package version 1.2.1]: https://github.com/matplotlib/basemap/releases

## Documentation and Usage

You can find the current TTLocVis master branch
documentation at our [documentation website].

[documentation website]: https://ttlocvis.readthedocs.io/en/latest/

## Community guidelines

Contributions to TTLocVis are welcome.

- Just file an Issue to ask questions, report bugs, or request new features.
- Pull requests via GitHub are also welcome.

Potential contributions include ways to further improve the quality of the LDA topics in handling the noisy
Twitter data and an improvement of the *loc_vis* method in a way that it becomes independent form the *basemap*
module.

## Authors

- Gillian Kant
- Christoph Weißer
- Benjamin Säfken

## License

TTLocVis is published under the __GNU GPLv3__ license.