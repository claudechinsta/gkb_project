# File Name:    gkb_textClf_w2v_main.py
# Author:       Boya Chen
# Created Date: 1 June 2017
# Description:  This is a python program that handle the machine learning for text classification
# using word-to-vectors.
# Python Version:   2.7.12
# External Python Packages:
# | scikit-learn [0.18.1]
# | nltk [3.2.2] (data required listed below, using nltk.download() to download uninstalled data )
# | --- nltk.stem.wordnet
# | --- nltk.word_tokenize
# | --- nltk.corpus.stopwords
# | gensim [2.0.0]

from collections import Counter, defaultdict
import nltk
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer.lemmatize
tokenizer = nltk.word_tokenize
stopwords = nltk.corpus.stopwords.words()
import csv
import random
import gensim, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from time import time

# Tokenise, lower and lemmatize a list of text, also remove stopswords
def clean_sentences(filePath, text):
    try:
        cleaned_sents = []
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                cleaned_sents.append(row)
        return cleaned_sents
    except IOError:
        print "Pre-processed file not exist\nStart 'cleaning' the sentences"
        corpus = []
        for line in text:
            if line == "":
                corpus.append([])
            line = tokenizer(line)
            line = [lemmatizer(token.lower()) for token in line if token.lower() not in stopwords]
            corpus.append(line)
        return corpus

#############################
# Building Words Model
# 1 - Data Preprocessing
#############################

print "[ 1 | Start Data Preprocessing ] ... ..."; t1 = time()

# Choose which label
# (0: geolocation related; 1: appearance related)
GEO = 0
APP = 1

TOPIC = GEO

DATASET_PATH = "./dataset.txt" # <-- The original Dataset
CLEANED_DATASET_PATH = "./cleaned_sents.csv" # <-- For developing, there is a pre-processed dataset that contain
                                             # tokenised and lowered sentences. (Optional Use)
POSTAG_DATASET_PATH = "./training_pos_tag.csv" # <-- Also for developing, there is a pre-processed dataset which
                                               # only contain POS tags for tokens of each sentences (Optional Use)
# | Reading the Dataset
content = []
with open(DATASET_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        content.append(row)


##########
# Labels
##########

# Acquire the list of labels
labels = []
for item in content:
    labels.append(item[TOPIC])

# Acquire a set of indexes of Positibe and Negative labelled 'docs'
indexes_of_true = [n for n, l in enumerate(labels) if l == 't']
indexes_of_false = [n for n, l in enumerate(labels) if l == 'f']

#############
# Sentences
#############

# Acquire raw sentences
sents = []
for item in content:
    sents.append(item[2])

# Reading Cleaned Sentences
cleaned_sents = clean_sentences(CLEANED_DATASET_PATH, sents)

# Acquire ent for each sentences
# ent_sents = []
# type_sents = []
# for item in content:
#     ent_sents.append(item[3])
#     try:
#         type_sents.append(ent_with_info[item[3]]['ent_type'])
#     except:
#         type_sents.append([])

############
# POS tags
############

# Acquire POS tags
pos_sents = []
with open(POSTAG_DATASET_PATH, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        pos_sents.append(row)

print '[ 1 | Finished ] --', round((time() - t1), 3), "s"
print

#########################
## 2| GENSIM W2V MODEL ##
#########################
print "[ 2 | Start Loading Gensim w2v model ] ... ..."; t1 = time()

# Corpus for Training
corpus = cleaned_sents[:]

# Reading Word2Vec Model or Re-train the Model
fname = 'wiki_corpus_300.model'

is_overwrite = 0
if os.path.exists(fname) and is_overwrite == 0:
    # load the file if it has already been trained, to save repeating the slow training step below
    model = gensim.models.Word2Vec.load(fname)
else:
    # can take a few minutes, grab a cuppa
    model = gensim.models.Word2Vec(corpus, size=300, min_count=1, workers=4, iter=50)
    model.save(fname)

print '[ 2 | Finished ] --', round((time() - t1), 3), "s"
print

######################################
## 3| APPLY GENSIM MODEL TO DATASET ##
######################################
print "[ 3 | Start Applying Gensim w2v model to Dataset ] ... ..."; t1 = time()

cut_num = 17000 # <-- Number of Labelled Docs, In this project, I only labelled 17000 lines for "GEO-related" sentences

words_with_t = []
for i, sent in enumerate(corpus):
    if labels[i] == 't':
        words_with_t.extend(sent)
words_with_t = list(set(words_with_t))

vector_model = model
dimen = vector_model.vector_size

corpus_matrix = []
for i, sent in enumerate(cleaned_sents[:cut_num]):
    # try:
        if i % 5000 == 0:
            print i, 'done.'
        if sent == []:
            corpus_matrix.append(np.zeros((1,dimen))[0])
            continue
        sent_matrix = np.array([vector_model[word] for word in sent if word in model])
        length = sent_matrix.shape[0]
        center = [sum(sent_matrix[:,i]) / length for i in range(dimen)]
        corpus_matrix.append(center)
    # except Error e:
    #     break;
        # corpus_matrix.append(np.zeros((1,dimen))[0])

# Vectorized Dataset for Training
t_dataset = np.array(corpus_matrix[:cut_num])
# Labels for Training
t_labels = labels[:cut_num]
# Vectorized Dataset for Prediction
# p_dataset = np.array(corpus_matrix[cut_num:])

print '[ 3 | Finished ] --', round((time() - t1), 3), "s"
print

##############################
# Dataset Imbalance Handling
##############################
indexes_of_true_r = indexes_of_true[:]
indexes_of_false_r = indexes_of_false[:]
random.shuffle(indexes_of_false_r)
indexes_of_false_r = indexes_of_false_r[:len(indexes_of_true_r) * 2]

indexes_of_r = list(sorted(indexes_of_true_r + indexes_of_false_r))
np.random.shuffle(indexes_of_r) # randomly shuffle the order

t_dataset_r = np.array([t_dataset[i] for i in range(len(t_labels)) if i in indexes_of_r])
t_labels_r = [t_labels[i] for i in range(len(t_labels)) if i in indexes_of_r]

print "size of positive training dataset:", len(indexes_of_true_r)
print "size of negative training dataset:", len(indexes_of_false_r)

# Visualise Data
def x_norm(array):
    return (array - min(array)) / (max(array) - min(array)) if (max(array) - min(array)) !=0 else array

def pca_visulisation(X, y, mode=0):
    # mode = 0(default): show all positive and negative points
    # mode = 1: Only show positive points
    # mode = 2: Only show negative points

    if X.shape[0] != len(y):
        return "[error]: X and y don't have same dimension"

    t_dataset_norm = np.array([x_norm(r) for r in X])

    pca = sklearnPCA(n_components=2)
    transformed = pca.fit_transform(t_dataset_norm)

    transformed_true = []
    transformed_false = []
    for i in range(len(transformed)):
        if y[i] == "t":
            transformed_true.append(list(transformed[i]))
        else:
            transformed_false.append(list(transformed[i]))
    transformed_true = np.array(transformed_true)
    transformed_false = np.array(transformed_false)

    color1, marker1 = "green", 'x'
    color2, marker2 = "blue", 'o'

    alpha1 = 0.2 if mode == 0 or mode == 1 else 0
    alpha2 = 0.2 if mode == 0 or mode == 2 else 1

    plt.scatter(transformed_false[:,0], transformed_false[:,1], label = "negative", c=color1, marker=marker1,
                alpha = alpha1)
    plt.scatter(transformed_true[:,0], transformed_true[:,1], label = "positive", c=color2, marker=marker2,
                alpha = alpha2)

    plt.legend()
    plt.show()

# pca_visulisation(t_dataset_r, t_labels_r)

##########################################
## 4. FITTING CLASSIFIER AND VALIDATION ##
##########################################
print "[ 4 | Fitting Classifiers and Validation ] ... ..."; t1 = time()

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report


def check_results(predictions, classifications):
    lab = ['t', 'f']
    #     print "accuracy"
    #     print accuracy_score(classifications, predictions)
    print classification_report(classifications, predictions, labels=lab)


############################
# Classifiers from Sklearn #
############################
# # Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()
# # Support Vector Machine
from sklearn import svm
clf_svm = svm.LinearSVC(C=0.1)
# # Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
# # Random Foreset Classifier
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier()
# # Bagging Classifier
from sklearn.ensemble import BaggingClassifier
clf_bag = BaggingClassifier()
# # K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
clf_knc = KNeighborsClassifier(n_neighbors=3)


# VALIDATING DIFFERENT CLASSIFIERS
# < -- Put classifiers in a list and validate them respectively
test_clf = [clf_svm, clf_rfc, clf_knc]

t1 = time()
print "[Job Start]"
for clf in test_clf:
    #     print 'CLASSIFIER -->', str(clf)
    #     print '------- CROSS VALIDATION --------'
    crossval_predicted = cross_val_predict(clf, t_dataset_r, t_labels_r, cv=20)
    check_results(crossval_predicted, t_labels_r)
    print '----------------------------------------------------'
    print '\n'

print "[Job Done]:", time() - t1, "s"




