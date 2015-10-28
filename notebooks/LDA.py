# Databricks notebook source exported at Wed, 28 Oct 2015 07:24:37 UTC
from pyspark.mllib.clustering import LDA
from pyspark.mllib.linalg import SparseVector

def runOnlineLDA(data, numOfTags, K = 10):
  '''
  require preprocessed input data
  input format:
    (key, [values])
  All values would be equally weighted.
  '''
  corpus = data.map(lambda (key, values): list(values)) \
               .filter(lambda p: len(p) >= 2) \
               .zipWithIndex().map(lambda (p, index): (index + 1, p)) \
               .mapValues(lambda p: SparseVector(numOfTags, {val: 1.0 for val in p})) \
               .map(lambda (index, values): [index, values]).cache()

  return LDA.train(corpus, K, optimizer='online')


def getTopics(ldaModel, valuesDict):
  topics = ldaModel.topicsMatrix()
  K = len(topics[0])
  print K
  topWords = [[] for k in range(K)]
  for word in range(0, ldaModel.vocabSize()):
    for topic in range(K):
      topWords[topic].append(topics[word][topic])
  for topic in range(K):
    topWords[topic] = np.array(topWords[topic])
    topWords[topic] = topWords[topic] / topWords[topic].sum()
    topWords[topic] = [(topWords[topic][w], valuesDict[w]) for w in range(topWords[topic].size)]
    topWords[topic].sort(reverse=True)
  return topWords

