# Databricks notebook source exported at Wed, 4 Nov 2015 19:31:20 UTC
import sys
from hashlib import md5

from numpy import log2
import numpy as np

MAXINT = sys.maxint
SEED = 20151028

class HashMD5:
  def __init__(self, count, length = 5):
    np.random.seed(SEED)
    self.count = count
    self.length = length
    self.addon = np.array([np.random.randint(MAXINT) for i in range(count)])

  def getHash(self, intVal, strVal = '', mod = None):
    length = self.length
    def hash(value):
      return int(md5((str(value) + strVal).encode('utf-8')).hexdigest()[-length:], 16)
    p = np.ones(self.count) * int(intVal) + self.addon
    values = np.fromiter(map(hash, p), dtype=np.int32)
    if mod:
      return values % mod
    return values

  def getCount(self):
    return self.count

def getInitial(length):
  minHash = np.ones(length) * np.inf
  tailZeros = np.zeros(length)
  return minHash, tailZeros

def countTailZeros(x):
  '''
  count x's trailing zero bits,
  so if x is 1101000 (base 2), return 3
  '''
  if x:
    c = 0
    x = (x ^ (x - 1)) >> 1 # Set x's trailing 0s to 1s and zero rest
    while x:
      c = c + 1
      x = (x >> 1)
    return c
  return np.inf

def getEstimate(hashVal):
  minHash = hashVal
  tailZeros = np.array([countTailZeros(x) for x in hashVal])
  return (minHash, tailZeros)

def reduceEstimates(p1, p2):
  minHash1, tailZeros1 = p1
  minHash2, tailZeros2 = p2
  return np.minimum(minHash1, minHash2), np.maximum(tailZeros1, tailZeros2)

def getUniqueItemsApproximate(tailZeros, numGroups, groupSize):
  d = tailZeros.reshape((numGroups, groupSize))
  means = np.mean(d, axis = 1)
  return 2**np.median(means)

def getSimilarity(minHash1, minHash2):
  return 1.0 * np.sum(minHash1 == minHash2) / minHash1.size

def getJSD(p1, p2, numGroups, groupSize):
  minHash1, tailZeros1 = p1
  minHash2, tailZeros2 = p2
  n1, n2 = getUniqueItemsApproximate(tailZeros1, numGroups, groupSize), getUniqueItemsApproximate(tailZeros2, numGroups, groupSize)
  p = getSimilarity(minHash1, minHash2)
  js10 = 1.0 / (2 * n1)
  js01 = 1.0 / (2 * n2)
  js11 = 1.0 / (2 * n1) * log2(2.0 * n2 / (n1 + n2)) + 1.0 / (2 * n2) * log2(2.0 * n1 / (n1 + n2))
  n10 = (n1 - p * n2) / (1 + p)
  n01 = (n2 - p * n1) / (1 + p)
  n11 = p * (n1 + n2) / (1 + p)
  return n10 * js10 + n01 * js01 + n11 * js11

def getMutualInformation(p1, p2, N, numGroups, groupSize):
  minHash1, tailZeros1 = p1
  minHash2, tailZeros2 = p2
  n1, n2 = getUniqueItemsApproximate(tailZeros1, numGroups, groupSize), getUniqueItemsApproximate(tailZeros2, numGroups, groupSize)
  p = getSimilarity(minHash1, minHash2)
  n11 = p * (n1 + n2) / (1 + p)
  return log2(n11 * N / (n1 * n2))

def getLogRatioDiff(pEntity, pCentroid, N, numGroups, groupSize, delta):
  minHashE, tailZerosE = pEntity
  minHashC, tailZerosC = pCentroid
  nE, nC = getUniqueItemsApproximate(tailZerosE, numGroups, groupSize), getUniqueItemsApproximate(tailZerosC, numGroups, groupSize)
  p = getSimilarity(minHashE, minHashC)
  n11 = p * (nE + nC) / (1 + p)
  eps = np.sqrt(log2(2.0 / delta) / (2.0 * nE))
  return log2(n11 / nE - eps) - log2(nC / N)
