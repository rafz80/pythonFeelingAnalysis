# -*- coding: utf-8 -*-
"""Sentiment analysis example by Raphael Ramos


Original file is located at
    https://github.com/rafz80
"""

import nltk

#nltk.download()

base = [('Bad conditions', 'positive'),
        ('Good', 'positive'),
        ('according', 'positive'),
        ('not ok', 'negative'),
        ('is ok', 'positive'),
        ('Satisfactory', 'positive'),
        ('Unsatisfactory', 'negative'),
        ('does not work', 'negative'),
        ('poor quality service', 'negative'),
        ('without accessibility', 'negative'),
        ('suitable', 'positive'),
        ('healthy', 'positive')]

stopwordsnltk = nltk.corpus.stopwords.words('english')

def fazstemmer(text):
    """
    Filter only the prefix of the words
    """
    stemmer = nltk.stem.RSLPStemmer()
    phrasessstemming = []
    for (words, emotion) in text:
        comstemming = [str(stemmer.stem(p))
                       for p in words.split() if p not in stopwordsnltk]
        phrasessstemming.append((comstemming, emotion))
    return phrasessstemming

phrasescomstemming = fazstemmer(base)

def searchwords(phrases):
    """
    looks the words in phrases and share emotions
    """
    allwords = []
    for (words, emotion) in phrases:
        allwords.extend(words)
    return allwords


words = searchwords(phrasescomstemming)

def searchFrequency(words):
    """
    Define the frequency that words occurs in database
    """
    words = nltk.FreqDist(words)
    return words

words = searchwords(phrasescomstemming)

trainingfrequency = searchFrequency(words)

def search_uniquewords(frequencia):
    """
    Create a unique words dictionary
    """
    freq = frequencia.keys()
    return freq

uniquewords = search_uniquewords(trainingfrequency)

def extractwords(document):
    doc = set(document)
    characteristics = {}
    for words in uniquewords:
        characteristics['%s' % words] = (words in doc)
    return characteristics


#typephrase = extractwords(['am', 'nov', 'dia'])
#print(typephrase)

completebase = nltk.classify.apply_features(extractwords, phrasescomstemming)

classification = nltk.NaiveBayesClassifier.train(completebase)


# the phrase that will be analised by algorithm
# ---------------------------------------------
# ---------------------------------------------
test = 'This product is ok!'
# ---------------------------------------------
# ---------------------------------------------
print('-----------------------')
print(test)


teststem = []
stemmer = nltk.stem.RSLPStemmer()
for (wordstraining) in test.split():
    comstem = [p for p in wordstraining.split()]
    teststem.append(str(stemmer.stem(comstem[0])))

new_phrase = extractwords(teststem)

distribution = classification.prob_classify(new_phrase)
print('-----------------------')
for classe in distribution.samples():
    print("%s: %f" % (classe, distribution.prob(classe)))
