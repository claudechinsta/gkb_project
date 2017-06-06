# 2017 Research Project: Geolocation Knowledge Base - Text Classification Source Code Documentation
Author: Boya Chen
Created At: 27 May 2017
Purpose: documentation of this project's source code

## 1. Introduction
This is the source code for 2017 Geolocation Knowledge Base - Text Classification. The purpose for the project is to predict the sentences relating to geolocation information from wikipedia articles. This repository mainly contain two parts

- Source Code for Machine Learning (./python_dev)
- Source Code for Front-end Development (./website_dev)

## 2. Files Description
### NodeJS enviornment
folder path: ./

| File Names | Description |
| --- | --- |
| package.json | Settings for NodeJS |
| node_modules | installed NodeJS modules |

### Source Code for Machine Learning (Python)
folder path: ./pathon_dev <br>
python enviroment: 2.7.10

| File Names | Description |
| --- | --- |
| dataset.txt | Wikipedia sentences from Wikipedia with two columns of label information and OSM IDs. There are total 4 columns and 93994 rows. The first column is labels for "Geolocation" topic and second one is for "Appearance". "t" means this sentence is informative, otherwise, the sentence is not relevant. The third column is the raw sentences. Last column is OSM ID for geolocation entities.|
| dataset_cleaned_sentences_93994.csv | For developing, I stored a pre-processed dataset. This dataset contain sentences from dataset.txt but already tokenised, converted to lowercases and lemmatised. Stop words have also been removed |
| dataset_pos_tags_93994.csv | Based on data from dataset_cleaned_sentences_93394.csv. I also saved a pre-processed dataset contain pos tags for each cleaned sentence in order to development. |
| entities_set.json | OpenStreetMap information for entities |
| ent_geojson.json | GEOJSON format data contains geographical inforamtion for entities |
| ent_geojson_3377_with_sents.json | Same like ent_geojson.json but contain topic-related sentences for each entity |
| GKB_Text_Classification_Hybrid.ipynb | The IPython notebook for experiments and results generation using hybrid features |
| GKB_Text_Classification_Word2vec.ipynb | The IPython notebook for experiments and results generation using word2vec as feature |
| wiki_corpus_300.model* | wiki_corpus_300.model, wiki_corpus_300.model.syn1neg.npy and wiki_corpus_300.model.wv.syn0.npy are model data pre-trained by Gensim for development |

### Source Code for Front-End Development
folder path: ./website_dev

| File Names | Description |
| --- | --- |
| index.html | Main applicaiton interface html page |
| css/ | This folder contains CSS stylesheets |
| imgs/ | This folder contains required images used in the interface |
| js/ | This folder contains all JavaScripts files including imported external resource like Bootstrp, JQuery and a customed file |
| js/mapboxControl.js | The javascript does data loading and control the events of this web app |

## 3. Installation an Usage

1. Install Node JS
    - Download: https://nodejs.org/en/
2. Install Canopy or Jupyter Notebook for opening .ipynb files. (if not installed)
    - Canopy: https://store.enthought.com/downloads/
    - Jupyter: http://jupyter.org/install.html
3. Install Required Python Packages (if not installed):
    - scikit-learn [0.18.1]
    - numpy [1.12.1] 
    - pandas [0.19.2] ; 
    - gensim [2.0.0] ; 
    - matplotlib [2.0.0]
4. To run a http-server, type the npm command in terminal(UNIX/MACOS) or command prompt(WIN) as root: <br/>
    
    `$ npm start` <br/>
    
    This command will run the server and automatically open application interface in the browser.

