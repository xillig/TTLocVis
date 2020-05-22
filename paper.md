---
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
  - Natural Language Processing
  - Machine Learning
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

The package TTLocVis provides a broad range of methods to generate, clean, analyze and visualize the contents of Twitter
data. TTLocVis enables the user to work with geo-spatial Twitter data and to generate topic distributions from Latent 
Dirichlet Allocation (LDA) Topic Models (!!cite Blei et al.) for geo-coded Tweets. As such, TTLocVis is an innovative 
tool to work with geo-coded text on a high geo-spatial resolution to analyse the public discourse on various topics in 
space and time. The package can be used for a broad range of applications for scientific research to gain insights into 
topics discussed on Twitter. For instance, the package could be used to analyse the public discourse on the COVID-19 
pandemic on Twitter in different countries and regions in the world over time. In particular, 
data from the recently provided COVID-19 stream by Twitter can be analysed to research the discussion about COVID-19 
on Twitter.^[https://developer.twitter.com/en/docs/labs/covid19-stream/overview]

In general, Topic Models are generative probabilistic models, that provide an insight into hidden information 
in large text corpora by estimating the underlying topics of the texts in an unsupervised manner. In Topic Models, 
each topic is a distribution over words that can be labeled by humans. For the purpose of labelling histograms and 
word clouds (for example see graph) provide helpful visualizations for the decision-making process of the user.

Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world.
Subsequently, the inherently noisy Twitter data can be cleaned, transformed and exported. 
In particular, TTLocVis enables the user to apply LDA Topic models on extremely sparse Twitter data by preparing 
the Tweets for LDA analysis by pooling Tweets by hashtags. The hashtags pooling is implemented with the specifically 
adjusted hashtag pooling algorithm from Mehrotra et. al. (2013). The goal of hashtag pooling is to supply the 
Topic Models with longer documents than just single Tweets to reduce the problems of Topic Models to process short 
and sparse texts. 

The pooling idea can be summarized into the following steps: Pool all Tweets by existing hashtags and check the 
similarity of an unlabeled tweet with all labeled Tweets (hashtag-pools). Subsequently, the unlabeled Tweets join the 
hashtag-pool with the highest cosine similarity value, if the value exceeds a certain threshold. This process is 
repeated for all unlabeled tweets. The described algorithm originating from Mehrotra et. al. (2013) was self-implemented
in the package as a parallelized function in order to speed up the heavy computational task that comes with it.

The resulting topic distributions that are computed with a LDA model that is trained on the pooled Tweets are 
substantially improved. When trained with sufficient data, clear topics can be generated and the shortcoming of 
LDAs with short and sparse text are minimized. 

TTLocVis provides options for automatized Topic Model parameter optimization. Furthermore, a distribution over 
topics is generated for each document. The distribution of topics over documents can be visualized with various 
plotting methods (For example see figure Word Cloud). The average prevalence of topics in the documents at each day can 
be plotted as a time series (For example see figure Time Series.), in order to visualize, how topics develop over time.
 
Above this, the spatial distribution of Tweets can be plotted on a world map, which automatically chooses an appropriate
part of the world, in order to visualise the chosen sample of Tweets. As part of the mapping process, each Tweet is 
classified by its most prevalent topic and colour coded (For example see figure Word Map 1 and figure World Map 2 for 
the spatial distribution of the same selected topics at different points in time.)
 
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

![Time Series.\label{fig:Time Series}](figures/time_series.pdf)

![Word Cloud.\label{fig:Word Cloud}](figures/word_cloud.pdf)

![World Map 1.\label{fig:test2}](figures/world_map1.png)

![World Map 2.\label{fig:test2}](figures/world_map2.png)

# References