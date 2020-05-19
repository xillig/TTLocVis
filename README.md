
# TTLocVis
TTLocVis: A Twitter Topic Location Visualization package
TESTTT

## Summary 

TTLocVis is a package providing a wide range of tools to analyze the contents of Twitter data. The user will be provided
with methods so they can collect Tweets (using a Twitter developer account), clean and transform them (also possible to
make use of the data for tasks exceeding the scope of this package), preparing them for LDA analysis by pooling them 
using a distinct algorithm to account for problems when it comes to LDA analysis of short, sparse and noisy text.
Furthermore, it provides options for automatized Topic Model parameter optimization to get the best results for the 
users data sets. The resulting tweets topic distributions can be visualized using several plotting methods, ranging 
from the topics itself to the change of topical prevalence over time to a spatial visualization of the topical
prevalence. There are methods provided to gain insights into to resulting data itself regarding specific words the user 
is interested in and their change in prevalence over time.     

## Statement of need

## How to cite 


## Installation

The package can be installed via *pip*:
```commandline
python pip install TTLocVis
```

### Windows

After successful insallation, the user must install the [*basemap* package] manually via *pip*:
```commandline
python -m pip install *your basemap wheel here*
```
__Note:__ The *cpXX* in the filenames refer to the python version you will use. An example for python 3.6. would be the file 
*basemap-1.2.1-cp36-cp36m-win_amd64.whl* Remember, TTLocVis is developed to run only on Python 3.6, 3.7 and 3.8.

[*basemap* package]: https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap

### Linux and iOS

Download [basemap package version 1.2.1] and install it accordingly.

[basemap package version 1.2.1]: https://github.com/matplotlib/basemap/releases

## Documentation and Usage

You can find the current TTLocVis master branch
documentation at our [docs website].

[docs website]: https://ttlocvis.readthedocs.io/en/latest/

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