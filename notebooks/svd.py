# Databricks notebook source exported at Wed, 13 Jan 2016 19:29:39 UTC
# MAGIC %run /Users/jalafate@ucsd.edu/estimate-online/vault

# COMMAND ----------

# MAGIC %run /Users/jalafate@ucsd.edu/estimate-online/twitter-utils

# COMMAND ----------

# MAGIC %run /Users/jalafate@ucsd.edu/estimate-online/estimate

# COMMAND ----------

import numpy as np
from operator import add
import gc

# COMMAND ----------

# MAGIC %md # Initial

# COMMAND ----------

files = [p.path for p in dbutils.fs.ls('/mnt/%s/data' % MOUNT_NAME)]
len(files)

# COMMAND ----------

# Parameters
# data range
#dataStDate = 10
#dataStep = 20
dataSt, dataEd = 57, 95 #dataStDate, dataStDate + dataStep

# threshold for minimum occurrences of data
minX = 1
minY = 1

# Hash functions
hashLength = 100
numGroups, groupSize = 10, 10

# COMMAND ----------

# load data
dataURL = ','.join(files[dataSt:dataEd])
data = tweetsFromFile(dataURL).cache() # all tweets with hashtags in JSON format
processedData = data.map(lambda p: (getUser(p), getHashtags(p))) \
                    .filter(lambda (user, hashtags): len(hashtags) == 1) \
                    .cache()

print 'Date range:'
print data.map(lambda p: (getDate(p), getDate(p))).reduce(lambda p1, p2: (min(p1[0], p2[0]), max(p1[1], p2[1])))

# matrix generated
pairs = processedData.reduceByKey(lambda a, b: set(a).union(set(b))) \
                     .filter(lambda (user, hashtags): len(hashtags) >= 2) \
                     .flatMap(lambda (user, hashtags): [(hashtag, user) for hashtag in hashtags]) \
                     .cache()

# COMMAND ----------

# MAGIC %md ## Select top 100 hashtags

# COMMAND ----------

TOP = 100

topX = pairs.map(lambda (x, y): (x, 1)) \
            .reduceByKey(add) \
            .sortBy(lambda (x, xcount): xcount, ascending = False) \
            .map(lambda (x, xcount): x) \
            .take(TOP)
topX = set(topX)

print 'Number of users mentioned more than 1 hashtags (accurate):', pairs.map(lambda (x, y): y).distinct().count() # 342923
pairs = pairs.filter(lambda (x, y): x in topX).cache()
print 'Number of users mentioned top %d hashtags (accurate):' % TOP, pairs.map(lambda (x, y): y).distinct().count() # 235546

# COMMAND ----------

# Construct estimation vectors
hashFunc = HashMD5(hashLength, SEED = 12334)
estPairs = pairs.map(lambda (x, y): (x, getEstimate(hashFunc.getHash(y)))).cache()

estAllUsers = estPairs.map(lambda (x, y): y).reduce(reduceEstimates)
numOfUsers = getUniqueItemsApproximate(estAllUsers[1], numGroups, groupSize)

# 0 stands for background set
# xInfo structure: (name, (gid, est))
xInfo = estPairs.reduceByKey(reduceEstimates) \
                .map(lambda (x, yest): (x, (yest, getUniqueItemsApproximate(yest[1], numGroups, groupSize)))) \
                .sortBy(lambda (x, (yest, count)): count, ascending = False) \
                .map(lambda (x, (yest, count)): (x, (0, yest))) \
                .cache()

print 'Number of users (estimation):', numOfUsers

# COMMAND ----------

print xInfo.map(lambda p: p[0]).collect()

# COMMAND ----------

# MAGIC %md # Algorithm

# COMMAND ----------

# MAGIC %md ## Find the SVD of the similarity matrix on the top 100 hashtags

# COMMAND ----------

x100 = xInfo.map(lambda (x, (gid, yest)): (x, yest)) \
            .collect()
lenX100 = len(x100)
lenX100

# COMMAND ----------

def jaccard(i, j):
  return getSimilarity(x100[i][1][0], x100[j][1][0])

def pmi(i, j):
  p = getSimilarity(x100[i][1][0], x100[j][1][0])
  nA = getUniqueItemsApproximate(x100[i][1][1], numGroups, groupSize)
  nB = getUniqueItemsApproximate(x100[j][1][1], numGroups, groupSize)
  intersection = p * (nA + nB) / (1 + p)
  return intersection / (nA * nB)

# COMMAND ----------

# Construct the similarity matrix, where m_ij is the jaccard similarity between i and j
initial, func = lambda i: 1.0, jaccard
# initial, func = lambda i: 1.0 / getUniqueItemsApproximate(x100[i][1][1], numGroups, groupSize), pmi
simMat = []
for i in range(lenX100):
  simMat.append([])
  for j in range(i):
    simMat[i].append(simMat[j][i])
  simMat[i].append(initial(i))
  for j in range(i + 1, lenX100):
    simMat[i].append(func(i, j))

# Find the SVD of simMat
from numpy.linalg import svd
U, s, V = np.linalg.svd(simMat, full_matrices=True)
U.shape, V.shape, s.shape

# Projection
K = 20
print np.sum(s[:K]) / np.sum(s)
print
S = np.zeros((s.shape[0], s.shape[0]))
S[:K, :K] = np.diag(s[:K])
lowRank = np.dot(np.dot(U, S), V)
np.allclose(lowRank, simMat)

# K-means
from scipy.cluster.vq import vq, kmeans, whiten
whitened = whiten(lowRank)
numOfCenters = 10
centroids, distortion = kmeans(whitened, numOfCenters)

# Print result
groups = [[] for i in range(numOfCenters)]
for wid, (p, raw) in enumerate(zip(whitened, x100)):
  d = [np.sum((p - centroids[i])**2) for i in range(numOfCenters)]
  groups[np.argmin(d)].append((np.min(d), raw[0], wid))

for i in range(numOfCenters):
  print 'Group %d (%d)' % (i, len(groups[i]))
  print '========='
  groups[i].sort(reverse = True)
  print ', '.join(map(lambda p: p[1], groups[i]))
  print

# COMMAND ----------

from matplotlib import pyplot as plt

plotMat = np.copy(simMat)

fig, ax = plt.subplots(2)
heatmap = []
heatmap.append(ax[0].pcolor(plotMat))

newMap = {}
newMat = [[-1.0 for i in range(100)] for j in range(100)]
newMatIdx = 0
for item in groups:
  for p in item:
    newMap[p[2]] = newMatIdx
    newMatIdx = newMatIdx + 1
for a in range(100):
  for b in range(100):
    newMat[newMap[a]][newMap[b]] = plotMat[a][b]

heatmap.append(ax[1].pcolor(np.array(newMat)))

cax = fig.add_axes([0.93, 0.1, 0.03, 0.8])
cbar = fig.colorbar(heatmap[0], cax=cax, ticks= np.arange(0.0, 1.1, 0.1), format='%.1f')
display(fig)

# COMMAND ----------

allitemfreq = []
for i in range(100):
  for j in range(i + 1, 100):
    x = simMat[i][j]
    allitemfreq.append(x)
fig, ax = plt.subplots(1)
histogram = ax.hist(allitemfreq, log = True, bins = 100)
display(fig)

# COMMAND ----------

# MAGIC %md Another metric
# MAGIC Piecewise mutual information like

# COMMAND ----------

# Construct the similarity matrix, where m_ij is the jaccard similarity between i and j
# initial, func = lambda i: 1.0, jaccard
initial, func = lambda i: 1.0 / getUniqueItemsApproximate(x100[i][1][1], numGroups, groupSize), pmi
simMat = []
for i in range(lenX100):
  simMat.append([])
  for j in range(i):
    simMat[i].append(simMat[j][i])
  simMat[i].append(initial(i))
  for j in range(i + 1, lenX100):
    simMat[i].append(func(i, j))

# Find the SVD of simMat
from numpy.linalg import svd
U, s, V = np.linalg.svd(simMat, full_matrices=True)
U.shape, V.shape, s.shape

# Projection
K = 20
print np.sum(s[:K]) / np.sum(s)
print
S = np.zeros((s.shape[0], s.shape[0]))
S[:K, :K] = np.diag(s[:K])
lowRank = np.dot(np.dot(U, S), V)
np.allclose(lowRank, simMat)

# K-means
from scipy.cluster.vq import vq, kmeans, whiten
whitened = whiten(lowRank)
numOfCenters = 4
centroids, distortion = kmeans(whitened, numOfCenters)

# Print result
groups = [[] for i in range(numOfCenters)]
for wid, (p, raw) in enumerate(zip(whitened, x100)):
  d = [np.sum((p - centroids[i])**2) for i in range(numOfCenters)]
  groups[np.argmin(d)].append((np.min(d), raw[0], wid))

for i in range(numOfCenters):
  print 'Group %d (%d)' % (i, len(groups[i]))
  print '========='
  groups[i].sort(reverse = True)
  print ', '.join(map(lambda p: p[1], groups[i]))
  print

# COMMAND ----------

from matplotlib import pyplot as plt

plotMat = np.copy(simMat)

fig, ax = plt.subplots(2)
heatmap = []
heatmap.append(ax[0].pcolor(plotMat))

newMap = {}
newMat = [[-1.0 for i in range(100)] for j in range(100)]
newMatIdx = 0
for item in groups:
  for p in item:
    newMap[p[2]] = newMatIdx
    newMatIdx = newMatIdx + 1
for a in range(100):
  for b in range(100):
    newMat[newMap[a]][newMap[b]] = plotMat[a][b]

heatmap.append(ax[1].pcolor(np.array(newMat)))

cax = fig.add_axes([0.93, 0.1, 0.03, 0.8])
cbar = fig.colorbar(heatmap[0], cax=cax, ticks= np.arange(0.0, 1.1, 0.1), format='%.1f')
display(fig)

# COMMAND ----------

allitemfreq = []
for i in range(100):
  for j in range(i + 1, 100):
    x = simMat[i][j]
    allitemfreq.append(x)
fig, ax = plt.subplots(1)
histogram = ax.hist(allitemfreq, log = True, bins = 100)
display(fig)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

