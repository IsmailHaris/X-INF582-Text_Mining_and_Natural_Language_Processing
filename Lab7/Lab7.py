#!/usr/bin/env python
# coding: utf-8

# Lab 7 - Report
# ===
# 
# **Antoine Carossio**
# 
# NB: PLEASE READ THE JUPYTER NOTEBOOK VERSION

# In[8]:


import nltk
#nltk.download('brown')
from nltk import FreqDist
from nltk.tag import HiddenMarkovModelTrainer
from nltk.corpus import brown
from nltk.probability import ConditionalProbDist, ConditionalFreqDist, LidstoneProbDist
from nltk.metrics.scores import precision, recall, f_measure

from tqdm import tqdm_notebook as tqdm
from itertools import product
import numpy as np

from viterbi import *


# # Helpers

# In[9]:


def to_ids(main_set, processed_set):
    """
    Get the IDs of the elements from the processed_set into the main_set as a list
    """
    return [main_set.index(element) for element in processed_set]

def lists_to_array(a):
    """
    Convert a list of lists to numpy array, padded with 0s if needed
    """
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))])
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b

import itertools
def flatten(list_of_lists):
    """
    Flatten at list of lists to a lists
    """
    return list(itertools.chain.from_iterable(list_of_lists))


# # 1. Building a simple HMM

# In[4]:


def extract_data(processed_corpus):
    """
    Extract words and tags form the corpus
    """
    all_tags = []
    all_words = []

    for sent in processed_corpus:
        for word, tag in sent:
            all_words.append(word)
            all_tags.append(tag)
    
    return all_words, all_tags

def lidstone_cond_freq(processed_freq, norm_len, k=.1):
    """
    Apply Lidstone to a ConditionalFreq() object
    """
    
    factory = lambda fd: LidstoneProbDist(fd, k, norm_len)
    return ConditionalProbDist(processed_freq, factory)


# In[5]:


corpus = brown.tagged_sents()
print(corpus)


# ## Question 1

# In[6]:


all_words, all_tags = extract_data(corpus)

Q = list(set(all_tags))
V = list(set(all_words))

n = len(Q)
m = len(V)

A  = np.zeros((n,n))
B  = np.zeros((n,m))
Pi = np.zeros(n)

print("Number of states: {}".format(n))


# ## Questions 2 & 3

# We divide the corpus into a training set and a testing set:

# In[7]:


training = corpus[:-10]
testing  = corpus[-10:]

words_train, tags_train = extract_data(training)


# ### Matrix $A$

# Get the frequency conditional of nltk bigrams, and use the lidstone smoothing (with k=0.1) to estimate the conditional probabilities:

# In[8]:


tag_bigrams_freq_train = ConditionalFreqDist(nltk.bigrams(tags_train))
tags_bigrams_prob_train = lidstone_cond_freq(tag_bigrams_freq_train, n)

print(list(tag_bigrams_freq_train["AP"].items())[:10]) # Print only the first 10 elements
print(tag_bigrams_freq_train["AP"].freq("NN"))
print(tags_bigrams_prob_train["AP"].prob("NN"))
# Note : # tag_bigrams_freq_train["AP"] == tags_bigrams_prob_train["AP"].freqdist()


# In[9]:


for i in range(n):
    for j in range(n):
        A[i][j] = tags_bigrams_prob_train[Q[i]].prob(Q[j])
print(sum(A[1,:]))


# ### Vector $\pi$

# In[10]:


tags_first_train = [sent[0][1] for sent in training]
tags_first_freq_train = FreqDist(tags_first_train)
tags_first_prob_train = LidstoneProbDist(tags_first_freq_train, .1, n)

print(list(tags_first_freq_train.items())[:10])

for i in range(n):
    Pi[i] = tags_first_prob_train.prob(Q[i])
print(sum(Pi))


# ### Matrix $B$

# In[11]:


observations_freq_train = ConditionalFreqDist(zip(words_train, tags_train))
observations_prob_train = lidstone_cond_freq(observations_freq_train, m)

for i in tqdm(range(n)):
    for j in range(m):
        B[i][j] = observations_prob_train[Q[i]].prob(V[j])
print(sum(B[1,:]))


# ## Question 4

# In[12]:


tags_true = []
tags_pred = []
scores    = []

for sent in tqdm(testing):
    word_list = to_ids(V, [word for word, _ in sent])
    tag_list  = to_ids(Q, [tag for _, tag in sent])
    tags_true.append(tag_list)

    predicted, score = viterbi((Pi,A,B), word_list)
    tags_pred.append(predicted)
    scores.append(score)


# In[13]:


predicted_set = set(flatten(tags_pred))
reference_set = set(flatten(tags_true))

print('Precision :', precision(predicted_set, reference_set))
print('Recall    :', recall(predicted_set, reference_set))
print('F1-score  :', f_measure(predicted_set, reference_set))


# ## Question 5

# We only conserve the pairs of tags which appears at least once in the whole set.

# In[13]:


Q_tri = list(set(nltk.bigrams(all_tags)))
n_tri = len(Q_tri)


# Recompute the Matrix $A$

# In[14]:


tag_trigrams_freq_train = ConditionalFreqDist(((w0, w1), w2) for w0, w1, w2 in nltk.trigrams(tags_train))
tags_trigrams_prob_train = lidstone_cond_freq(tag_trigrams_freq_train, n)
tags_trigrams_prob_train[('AT', 'NP-TL')].freqdist()


# In[15]:


A_tri = np.zeros((n_tri,n))

for i in tqdm(range(n_tri)):
    for j in range(n):
        A_tri[i][j] = tags_trigrams_prob_train[Q_tri[i]].prob(Q[j])
print(sum(A_tri[1,:]))


# In[16]:


tags_true_tri = []
tags_pred_tri = []
scores_tri    = []

for sent in tqdm(testing):
    word_list = to_ids(V, [word for word, _ in sent])
    tag_list  = to_ids(Q, [tag for _, tag in sent])
    tags_true_tri.append(tag_list)

    predicted, score = viterbi((Pi,A_tri,B), word_list)
    tags_pred_tri.append(predicted)
    scores_tri.append(score)


# In[17]:


predicted_set_tri = set(flatten(tags_pred_tri))
reference_set_tri = set(flatten(tags_true_tri))

print('Precision :', precision(predicted_set_tri, reference_set_tri))
print('Recall    :', recall(predicted_set_tri, reference_set_tri))
print('F1-score  :', f_measure(predicted_set_tri, reference_set_tri))


# # 2. Using NLTK’s HMM implementation

# ### Question 6

# In[35]:


tags_train_tri = list(set(nltk.bigrams(tags_train)))
observations_freq_train_tri = ConditionalFreqDist(zip(words_train, tags_train_tri))
observations_prob_train_tri = lidstone_cond_freq(observations_freq_train_tri, m)


# In[139]:


trainer = HiddenMarkovModelTrainer(all_tags, all_words)
hmm = trainer.train_supervised(training, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))


# In[141]:


tags_true_lib = []
tags_pred_lib = []

for sent in tqdm(testing):
    word_list = [word for word, _ in sent]
    tag_list  = [tag for _, tag in sent]
    tags_true_lib.append(tag_list)

    tag_pred = hmm.tag(word_list)
    tags_pred_lib.append(el[1] for el in tag_pred)


# In[142]:


predicted_set_lib = set(flatten(tags_pred_lib))
reference_set_lib = set(flatten(tags_true_lib))

print('Precision :', precision(predicted_set_lib, reference_set_lib))
print('Recall    :', recall(predicted_set_lib, reference_set_lib))
print('F1-score  :', f_measure(predicted_set_lib, reference_set_lib))


# As we could have guessed easily it is better to used the HMM of the NLTK library

# # 3. NER using Conditional Random Fields

# In[2]:


import os
import pandas as pd

from pprint import pprint
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report

from crf_helper import *


# In[3]:


data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill") #repeat sentence number on each row

words = list(set(data["Word"].values)) #vocabulary V
tags = list(set(data["Tag"].values)) #vocabulary V
n_words = len(words)
n_tags = len(tags)


# In[4]:


getter = SentenceGetter(data) #transform sentences into sequences of (Word, POS, Tag)
sentences = getter.sentences

pprint(sentences[0][:5])


# In[5]:


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

pprint(X[0][:3])
pprint(y[0][:3])


# In[17]:


crf = CRF(algorithm='lbfgs', max_iterations=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)
crf.fit(X_train, y_train)
pred = crf.predict(X_test)
report = flat_classification_report(y_pred = pred, y_true = y_test)
print(report)


# ## Question 7

# ### Part a) CRF Equivalent to bigram HHM

# In[ ]:


X_hmm = []
hmm_features = ["-1:postag", "postag"]

for sent in tqdm(X):
    sent_words = []
    for word in sent:
        word_hmm_features = dict()
        for feature in hmm_features:
            if feature in word:
                word_hmm_features[feature] = word[feature]
        sent_words.append(word_hmm_features)
    X_hmm.append(sent_words)
    
pprint(X_hmm[0])


# In[32]:


crf_hmm = CRF(algorithm='lbfgs', max_iterations=100)
X_train, X_test, y_train, y_test = train_test_split(X_hmm, y, test_size=0.33, random_state=22)
crf_hmm.fit(X_train, y_train)
pred = crf_hmm.predict(X_test)
report = flat_classification_report(y_pred = pred, y_true = y_test)
print(report)


# ### Part b) Features to improve the results for PER, GEO and ORG

# First let's take a look of words tagged by `B-geo`, `B-org`, `B-per`, `I-geo`, `I-org` and `I-per`

# In[6]:


def get_context(label):
    all_words = flatten(sentences)
    res = []
    for i in range(1,len(all_words)-1):
        if all_words[i][2] == label:
            res.append([all_words[i-1], all_words[i], all_words[i+1]])
    
    return res


# In[10]:


Bgeo_sents = get_context("B-geo") # Geographical locations
Borg_sents = get_context("B-org") # Organisations
Bper_sents = get_context("B-per") # Persons
Igeo_sents = get_context("I-geo") # Geographical locations
Iorg_sents = get_context("I-org") # Organisations
Iper_sents = get_context("I-per") # Persons


# In[11]:


pprint(Bgeo_sents[:10])
print()
pprint(Borg_sents[:10])
print()
pprint(Bper_sents[:10])
print()
pprint(Igeo_sents[:10])
print()
pprint(Iorg_sents[:10])
print()
pprint(Iper_sents[:10])


# By looking at some samples of each tags, we can see that some new features be could revelant to add, such ad:
# - Does the word starts with a capital letter?
# - Does the previous word starts with a capital letter? (especially for `B-per`)
# - What is the total number of capitals in the word?
# - What is the total number of dots in the word ?

# In[18]:


def starts_with_capital(word):
    return int(word[0].isupper())

def number_of_capitals(word):
    return sum(letter.isupper() for letter in word)

def number_of_dots(word):
    return word.count(".")

def word2features_plus(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = { #features related to the current position
        'bias': 1.0,
        'word.lower()': word.lower(),
        'postag': postag,
    }
    
    ## +1 and +2 words
    if i > 0: #features related to preceding word/tag
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:postag': postag1,
            '-1:starts_with_capital': number_of_capitals(word1)       # NEW
        })
        if i > 1:
            word2 = sent[i-2][0]
            postag2 = sent[i-2][1]
            features.update({
                '-2:word.lower()': word2.lower(),                     # NEW
                '-2:start_with_capital' : starts_with_capital(word2), # NEW
                '-2:postag': postag2,                                 # NEW
            })
    else:
        features['BOS'] = True #feature for Beginning of Sentence

    ## -1 and -2 words
    if i < len(sent)-1: #features related to the following word/tag
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:postag': postag1,
            '+1:start_with_capital' : starts_with_capital(word1),     # NEW
        })
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            postag2 = sent[i+2][1]
            features.update({
                '+2:word.lower()': word2.lower(),                     # NEW
                '+2:postag': postag2,                                 # NEW
                '+2:start_with_capital' : starts_with_capital(word2), # NEW
            })
    else:
        features['EOS'] = True #feature for end of sentence

    features.update({
        'starts_with_capital': starts_with_capital(word),             # NEW
        'number_of_capitals': number_of_capitals(word),               # NEW
        'number_of_dots': number_of_dots(word)                        # NEW
    })

    return features

#transform the sentence in a sequence of features
def sent2features_plus(sent):
    return [word2features_plus(sent, i) for i in range(len(sent))]


# In[19]:


X_plus = [sent2features_plus(s) for s in tqdm(sentences)]


# In[20]:


crf_plus = CRF(algorithm='lbfgs', max_iterations=100)
X_train, X_test, y_train, y_test = train_test_split(X_plus, y, test_size=0.33, random_state=22)
crf_plus.fit(X_train, y_train)
pred = crf_plus.predict(X_test)
report = flat_classification_report(y_pred = pred, y_true = y_test)
print(report)


# ### Reports comparison

# With 11 new features the F1-score is improved for all tags, and especially for the tags of interests. The results are summarized in the following table:
# 
# | Tag   | HHM equivalent | Given CRF | CRF+ 11 features |
# |--|--|--|--|
# | B-geo | 0.67 | 0.86 | 0.87 |
# | B-per | 0.60 | 0.80 | 0.81 |
# | B-org | 0.35 | 0.73 | 0.75 |
# | I-geo | 0.42 | 0.78 | 0.79 |
# | I-per | 0.69 | 0.86 | 0.87 |
# | I-org | 0.50 | 0.77 | 0.79 |

# ## Question 8

# Tutorial: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# In[59]:


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss # Change n°0 (to avoid a Deprecation Warning)


# In[60]:


glove_dir = 'glove/' 
embeddings_index = {}

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# In[83]:


embedding_dim = 100 #if you use other embeddings, introduce the right size here
max_words = n_words+1 # Change n°1

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word2idx.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# In[84]:


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx  = {t: i for i, t in enumerate(tags)}


# In[85]:


max_len=75
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[86]:


y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)


# In[89]:


input_layer = Input(shape=(max_len,))
model = Embedding(max_words, embedding_dim, input_length=max_len, mask_zero=True, weights=[embedding_matrix], trainable=False)(input_layer) # Change n°2
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model) # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(n_tags)
out = crf(model)

model = Model(input_layer, out)
model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf.accuracy])  # Change n°0 (to avoid a Deprecation Warning)


# In[90]:


history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.33, verbose=1)
test_pred = model.predict(X_test, verbose=1)


# Results without Embedding:
# 
# ```bash=
# Train on 21528 samples, validate on 10604 samples
# Epoch 1/5
# 21528/21528 [==============================] - 86s 4ms/step - loss: 0.1555 - crf_viterbi_accuracy: 0.9598 - val_loss: 0.0747 - val_crf_viterbi_accuracy: 0.9761
# Epoch 2/5
# 21528/21528 [==============================] - 83s 4ms/step - loss: 0.0513 - crf_viterbi_accuracy: 0.9830 - val_loss: 0.0397 - val_crf_viterbi_accuracy: 0.9860
# Epoch 3/5
# 21528/21528 [==============================] - 83s 4ms/step - loss: 0.0315 - crf_viterbi_accuracy: 0.9885 - val_loss: 0.0311 - val_crf_viterbi_accuracy: 0.9882
# Epoch 4/5
# 21528/21528 [==============================] - 83s 4ms/step - loss: 0.0252 - crf_viterbi_accuracy: 0.9903 - val_loss: 0.0287 - val_crf_viterbi_accuracy: 0.9882
# Epoch 5/5
# 21528/21528 [==============================] - 83s 4ms/step - loss: 0.0220 - crf_viterbi_accuracy: 0.9913 - val_loss: 0.0265 - val_crf_viterbi_accuracy: 0.9893
# 15827/15827 [==============================] - 14s 871us/step
# ```
