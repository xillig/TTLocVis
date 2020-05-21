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

The package TTLocVis provides a broad range of methods to generate, clean, analyze and visualize the content of Twitter data.
TTLocVis enables the user to work with geo-spatial Twitter data and to generate topic distributions from LDA topic models (cite) 
for geo-coded Tweets. As such, TTLocVis is an innovative tool to work with geo-coded text on a high geo-spatial resolution to 
analyse the public discourse on various topics in space and time for any location in the world. As such, the package has
a broad range of applications which are not only limited to scientific research. For instance, the package could be used
to analyse the public discourse on the COVID-19 pandemic on Twitter in different countries and regions in the world over time.

(In particular,  the SARS-CoV-2 Twitter data set that was made available by Twitter recently (link) to ).
  
Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world that 
is specified with its longitude and latitude. Subsequently, the inherently messy Twitter data can be cleaned, transformed and exported. 

In particular, TTLocVis enables the user to apply LDA Topic models on extremely sparse Twitter data by preparing the Tweets 
for LDA analysis hy pooling Tweets by Hashtags using cosine similarity to create longer pseudo-documents for better 
LDA estimations. The pooling is implemented with the the specifically adjusted Hashtag pooling algorithm (cite). 

Additionally, it provides options for automatized Topic Model parameter optimization. Topic models provide an insight in hidden information of large text data sets by generation underlying topic of the texts.
Each topic is a distribution over words (see graph) that can be labeled. Additionally, a distribution over topics is generated for
each document (see graphs). The distributions from the LDA model can be visualized with various plotting methods.

Ranging from the topics itself to the change of topical prevalence over time to a spatial visualization of the topical
prevalence. 

- histogramm 

There are methods provided to gain insights into to resulting data itself regarding specific words the user 
is interested in and their change in prevalence over time.   

During the described analysis the data can be easily exported and an. 

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