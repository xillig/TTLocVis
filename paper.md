title: 'TTLocVis: A Twitter Topic Location Visualization package'
tags:
  - Python
  - Twitter
  - Topic Modelling
  - LDA
  - Latent Dirichlet Allocation
  - Visualization
  - Spatial Modelling
  - Geocoded Text
  - Hashtag-Pooling Algorithm
authors:
  - name: Gillian Kant
    orcid: 0000-0003-2346-2841
    affiliation: 1
  - name: Christoph Weißer
    orcid: 0000-0003-0616-1027
    affiliation: 1
  - name: Benjamin Säfken
    orcid: 0000-0003-4702-3333
    affiliation: 1


affiliations:
 - name: Center of Statistics, Georg-August-Universität Göttingen, Germany
   index: 1
date: 25 May 2020
bibliography: paper.bib

---

# Summary

-- Mention of R-LDA package!

TTLocVis is a package providing a wide range of tools to analyze the contents of Twitter data. The user will be provided
with methods so they can collect Tweets (using a Twitter developer account), clean and transform them (also possible to
make use of the data for tasks exceeding the scope of this package), preparing them for LDA analysis by pooling them 
using a distinct algorithm to account for problems when it comes to LDA analysis of short, sparse and noisy text.
Furthermore, it provides options for automatized Topic Model parameter optimization to get the best results for the 
users data sets. The resulting tweets topic distributions can be visualized using several plotting methods, ranging 
from the topics itself to the change of topical prevalence over time to a spatial visualization of the topical
prevalence. There are methods provided to gain insights into to resulting data itself regarding specific words the user 
is interested in and their change in prevalence over time.   

The package provides a ordered working scheme, which provides the user with the ability to start collecting their own
tweets and processing them to use them accordingly or for other purposes. The cleaned tweets are then further prepared
by pooling. 
The pooling of tweets by hashtag to create pseudo-documents which will be fed into LDAs is a vital part of this package.
The goal of this preparation is to supply the Topic Models with longer documents than just single tweets to counteract
the problems (ZITIEREN) of Topic Models with short and sparse text. The pooling idea arose from Mehrotra et. al. (2013)
and is described as follows: 

- Pooling of all tweets by existing hashtags. 
- Check the similarity of an unlabeled tweet with all labeled ones (hashtag-pools).
- The unlabeled tweet joins the hashtag-pool with the highest cosine similarity value if the value exceeds a certain
threshold *C*.
- Repeat that procedure for all unlabeled tweets.   

The chosen measure for cosine similarity is TF-IDF. The authors show that hashtag-pooled tweets perform best, compared
to unpooled, author (i.e. user)-pooled and time-pooled documents (Mehrotra et. al. 2013, 892). The algorithm described 
above was self-implemented in the package as a parallelized function to speed up the heavy computational task that 
comes with it.

Feeding this data into the LDAs for training benefits the performance of the Topic Models quiet well. The resulting 
topic distributions for single tweets do resemble the topical prevalence way better when an LDA is trained without 
pooling. When trained with sufficient data, decent topics can be generated and the short coming of LDAs with short 
and sparse text can be overcome. The resulting topic distributions are used for gaining insight into the time- and 
spatial variation of topics. Our package provides the methods to visualize these information.  












































The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References