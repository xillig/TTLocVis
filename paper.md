---
title: 'TTLocVis: A Twitter Topic Location Visualization Package'
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
  - name: Christoph Weisser
    orcid: 0000-0003-0616-1027
    affiliation: 1
  - name: Benjamin Säfken
    orcid: 0000-0003-4702-3333
    affiliation: 1


affiliations:
 - name: Centre for Statistics, Georg-August-Universität Göttingen, Germany
   index: 1
date: 25 May 2020
bibliography: paper.bib

---

# Summary

The package TTLocVis provides a broad range of methods to generate, clean, analyze and visualize the contents of Twitter
data. TTLocVis enables the user to work with geo-spatial Twitter data and to generate topic distributions from Latent 
Dirichlet Allocation (LDA) Topic Models [@blei] for geo-coded Tweets. As such, TTLocVis is an innovative 
tool to work with geo-coded text on a high geo-spatial resolution to analyse the public discourse on various topics in 
space and time. The package can be used for a broad range of applications for scientific research to gain insights into 
topics discussed on Twitter. For instance, the package could be used to analyse the public discourse on the COVID-19 
pandemic on Twitter in different countries and regions in the world over time. In particular, 
data from the recently provided COVID-19 stream by Twitter can be analysed to research the discussion about COVID-19 
on Twitter.^[https://developer.twitter.com/en/docs/labs/covid19-stream/overview]

In general, Topic Models are generative probabilistic models, that provide an insight into hidden information 
in large text corpora by estimating the underlying topics of the texts in an unsupervised manner. In Topic Models, 
each topic is a distribution over words that can be labeled by humans. For the purpose of labelling histograms and 
word clouds (for example see graph) provide helpful visualizations for the decision-making process of the user [@blei].

Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world.
Subsequently, the inherently noisy Twitter data can be cleaned, transformed and exported. 
In particular, TTLocVis enables the user to apply LDA Topic Models on extremely sparse Twitter data by preparing 
the Tweets for LDA analysis by the pooling Tweets by hashtags. The hashtags pooling algorithm [@Mehrotra] is implemented 
in a parallelized form in order to speed up the heavy computational task. The goal of hashtag pooling is to supply the 
Topic Models with longer documents than just single Tweets to reduce the problems of Topic Models to process short and 
sparse texts. The pooling idea can be summarized into the following steps: Pool all Tweets by existing hashtags and 
check the similarity of an unlabeled Tweet with all labeled Tweets (hashtag-pools). Subsequently, the unlabeled Tweets
join the hashtag-pool with the highest cosine similarity value, if the value exceeds a certain threshold. This process is 
repeated for all unlabeled Tweets. The resulting topic distributions that are computed with a LDA model that is trained 
on the pooled Tweets are substantially improved. When trained with sufficient data, clear topics can be generated and 
the shortcoming of LDAs with short and sparse text are minimized. 

TTLocVis provides options for automatized Topic Model parameter optimization. Furthermore, a distribution over 
topics is generated for each document. The distribution of topics over documents can be visualized with various 
plotting methods (for example see figure Word Cloud). The average prevalence of topics in the documents at each day can 
be plotted as a time series (for example see figure Time Series), in order to visualize, how topics develop over time.
 
Above this, the spatial distribution of Tweets can be plotted on a world map, which automatically chooses an appropriate
part of the world, in order to visualise the chosen sample of Tweets. As part of the mapping process, each Tweet is 
classified by its most prevalent topic and colour coded (for example see figure Word Map 1 and figure World Map 2 for 
the spatial distribution of the same selected topics at different points in time).
 
# Figures


![Time Series.\label{fig:Time Series}](figures/time_series.pdf){ width=80% }

![Word Cloud.\label{fig:Word Cloud}](figures/word_cloud.pdf){ width=80% }

![World Map 1.\label{fig:test2}](figures/world_map1.png){ width=100% }

![World Map 2.\label{fig:test2}](figures/world_map2.png){ width=80% }

# References