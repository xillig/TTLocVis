#Usage and Examples

This section provides a few coding examples applying the functionalities of the TTLocVis package. To get to know the
Package, it is advised to use the JSON *example_files* provided in the *github repository* as data input for the 
classes *Cleaner* and *LDAAnalyzer*, skipping the streaming at first. __Note that the provided example data is for
presentation purposes only and does not provide any meaningful results!__   

After [installation], import the necessary classes.
```python
from TTLocVis.module import TwitterStreamer
from TTLocVis.module import Cleaner
from TTLocVis.module import LDAAnalyzer
```
[installation]: README.md

##Streaming
The Twitter data can be accessed easily when a [Twitter developer account] has been created by just one statement:
```python
TwitterStreamer(auth_path=r'path-to-Twitter API Access.txt',
                save_path=r'path-to-save-location', languages=['en'],locations=[-125,25,-65,48])
```
In this case, english-language tweets from a bounding-box roughly covering the US are collected via the `languages` 
and `locations` argument. A tool to get bounding-box coordinates can be found [here].
 
__Note:__ The streamer collects data until an error from the API occurs or the user performs a *KeyboardInterrupt* 
(Ctrl+C) 

[here]: https://boundingbox.klokantech.com/
[Twitter developer account]: https://developer.twitter.com/en

##Cleaning
Cleaning and transforming the raw JSON files is as simple as the streaming:
```python
c = Cleaner(load_path=r'path-to-a-JSON-file-folder')
print(c.raw_data)
c.saving(r'path-where-the-cleaned-data-shall-be-saved')
```
The resulting data is printed as it is saved in `self.raw_data`.

##





