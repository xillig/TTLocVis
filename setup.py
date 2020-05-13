import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

#version = str(sys.version_info[0]) + str(sys.version_info[1])
#basemap_version_links = {'36': 'https://www.lfd.uci.edu/~gohlke/pythonlibs/basemap-1.2.1-cp36-cp36m-win_amd64.whl',
#                         '37': 'test',
#                         '38': 'test',
#                         '39': 'test'}

installation_dependencies = [
    'pip>=19.1.1',
    #'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz',
    'gensim>=3.8.1',
    'matplotlib>=2.2.2',
    'numpy==1.16.1',
    'pandas==0.24.2',
    'pyproj>=2.6.1.post1',
    'scikit-learn>=0.21.2',
    'spacy>=2.2.2',
    'tweepy>=3.8.0',
    'urllib3>=1.25.8',
    'wordcloud>=1.7.0'
]

setuptools.setup(
    name="TTLocVis",
    version="0.0.1",
    author="Gillian Kant, Christoph Weisser, Benjamin Saefken",
    author_email="gilliankant@googlemail.com, c.weisser@stud.uni-goettingen.de, "
                 "benjamin.saefken@uni-goettingen.de",
    description="TTLocVis: A Twitter Topic Location Visualization package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xillig/TTLocVis",
    install_requires=installation_dependencies,
    dependency_links=installation_dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
