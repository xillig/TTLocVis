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
to analyse the public discourse on the COVID-19 pandemic on Twitter in different countries and regions in the world over time. In particular, 
data from the recently provided COVID-19 stream by Twitter can be analysed to research the discussion of COVID-19. The package might be 
for instance useful to research the spread of the virus or the dissemination of misleading information 
(https://developer.twitter.com/en/docs/labs/covid19-stream/overview). 

Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world that 
is specified with its longitude and latitude. Subsequently, the inherently messy Twitter data can be cleaned, transformed and exported. 

In particular, TTLocVis enables the user to apply LDA Topic models on extremely sparse Twitter data by preparing the Tweets 
for LDA analysis hy pooling Tweets by Hashtags using cosine similarity to create longer pseudo-documents for better 
LDA estimations. 

The pooling is implemented with the the specifically adjusted Hashtag pooling algorithm. 
The goal of this preparation is to supply the Topic Models with longer documents than just single tweets to counteract
the problems of Topic Models with short and sparse text. The pooling idea arose from Mehrotra et. al. (2013)
and is described as follows: Pool all tweets by existing hashtags and check the similarity of an unlabeled tweet with all labeled ones 
(hashtag-pools). Subsequently, the unlabeled join the hashtag-pool with the highest cosine similarity value, if the value exceeds a certain
threshold. This process is repeated for all unlabeled tweets.  The chosen measure for cosine similarity is TF-IDF. 
The described algorithm by Mehrotra et. al. (2013) was self-implemented in the package as a parallelized function 
in order to speed up the heavy computational task that comes with it.

The resulting topic distributions for which are computed with the LDA model that are trained on the pooled Tweets are substantially
improved. When trained with sufficient data, clear topics can be generated and the short coming of LDAs with short 
and sparse text is minimised. 

Additionally, it provides options for automatized Topic Model parameter optimization. Topic models provide an insight in hidden information of large text data sets by generation underlying topic of the texts.
Each topic is a distribution over words that can be labeled. For the the labelling histograms or wordclouds (for example see graph)
can be used. 
 
Additionally, a distribution over topics is generated for each document. The distribution of topics over documents
can be visualized with various plotting methods. The average prevalence of topics in the documents at each day can be plotted 
as a time series in order to visualise how topics develop over time (see graph). Above this, the spatial distribution of Tweets 
can be plotted on a map which automatically chooses an appropriate part of the world map to visualise the choosen sample of Tweets.
In the map the most prevalent topics in each Tweet are visualised with different colors (see graph). 

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


# References

