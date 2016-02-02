# Databricks notebook source exported at Tue, 2 Feb 2016 00:51:09 UTC
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

# MAGIC %md Don't run this

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

processedData.count()

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

pairs.repartition(1).saveAsPickleFile('/mnt/%s/pairs-1hashtag' % MOUNT_NAME)

# COMMAND ----------

# MAGIC %md Start from here

# COMMAND ----------

loadpairs = sc.pickleFile('/mnt/%s/pairs-1hashtag' % MOUNT_NAME)
print loadpairs.count() #, pairs.count() # 713703
pairs = loadpairs

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

from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt

def runKmeans(data, numOfCenters, seed = None, runs = 1, maxIterations = 100):
  # Build the model (cluster the data)
  clusters = KMeans.train(data.map(lambda (x, vec): vec), numOfCenters,
                          runs = runs, seed = seed, maxIterations = maxIterations)
  return clusters

def runAssignment(data, clusters, diffFunc = None):
  def getDists(point, clusters):
    f = lambda center: sqrt(sum([x**2 for x in (point - center)]))
    dists = map(f, clusters.centers)
    return dists
  if diffFunc is None:
     diffFunc = lambda p0, p1: p1 - p0
  second = 1
  if len(clusters.centers) == 1:
    second = 0
  assign = data.mapValues(lambda vec: getDists(vec, clusters)) \
               .mapValues(lambda dists: (np.argmin(dists), sorted(dists)[:2])) \
               .mapValues(lambda (gid, dists2): (gid, dists2[0], diffFunc(dists2[0], dists2[second])))
  return assign

def getKmeansGroup(assign, func = None):
  def getClass(ddiff, gid, func):
    if func(ddiff):
      return gid
    return -1
  if func is None:
    func = lambda k: True
  groups = assign.mapValues(lambda (gid, d, ddiff): getClass(ddiff, gid, func)) \
                 .map(lambda (x, gid): (gid, x)) \
                 .groupByKey() \
                 .sortByKey() \
                 .map(lambda (gid, g): g) \
                 .collect()
  return groups

def getKmeansMap(assign, outlier = -1, func = None):
  def getClass(ddiff, gid, func):
    if func(ddiff):
      return gid
    return -1
  if func is None:
    func = lambda k: True
  mapping = assign.mapValues(lambda (gid, d, ddiff): getClass(ddiff, gid, func)) \
                  .collect()
  return dict(mapping)

def getPartition(numOfCentersX, numOfCentersY, runs = 1, maxIterations = 100, diffFunc = None, SEED = 160119):
  def get_vector(v, length):
    v = dict(v)
    r = np.array([v.get(k, 0) for k in range(length)])
    return 1.0 * r / r.sum()

  xcluster = runKmeans(xvecs, numOfCentersX, runs = runs, maxIterations = maxIterations, seed = SEED)
  xassign = runAssignment(xvecs, xcluster)
  xMap = getKmeansMap(xassign)
  ylength = numOfCentersX
  yvecs = pairs.map(lambda (x, y): ((y, xMap[x]), 1)) \
               .reduceByKey(add) \
               .map(lambda ((y, xg), count): (y, (xg, count))) \
               .groupByKey() \
               .map(lambda (y, v): (y, get_vector(v, ylength))) \
               .sortByKey(numPartitions = 1) \
               .cache()
  ycluster = runKmeans(yvecs, numOfCentersY, runs = runs, maxIterations = maxIterations, seed = SEED)
  yassign = runAssignment(yvecs, ycluster, diffFunc)
  return (xcluster, xassign), (ycluster, yassign)

#################################

def jaccard(i, j):
  return getSimilarity(x100[i][1][0], x100[j][1][0])

def pmi(i, j):
  p = getSimilarity(x100[i][1][0], x100[j][1][0])
  nA = getUniqueItemsApproximate(x100[i][1][1], numGroups, groupSize)
  nB = getUniqueItemsApproximate(x100[j][1][1], numGroups, groupSize)
  intersection = p * (nA + nB) / (1 + p)
  return intersection / (nA * nB)

# COMMAND ----------

x100 = xInfo.map(lambda (x, (gid, yest)): (x, yest)) \
            .collect()
# .filter(lambda (x, (gid, yest)): x not in ['039s', '039certain', 'movember']) \
# pairs = pairs.filter(lambda (x, y): x not in ['039s', '039certain', 'movember']).cache()
reverseX100 = {p[0]: idx for idx, p in enumerate(x100)}
lenX100 = len(x100)
lenX100

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
# U.shape, V.shape, s.shape

# Projection
K = 20
print np.sum(s[:K]) / np.sum(s)
#'''
S = np.zeros((s.shape[0], s.shape[0]))
S[:K, :K] = np.diag(s[:K])
lowRank = np.dot(np.dot(U, S), V)
np.allclose(lowRank, simMat)
'''
Vk = V[:K, :]
lowRank = np.dot(simMat, Vk.transpose())
'''
print lowRank.shape
REMAIN = lowRank.shape[0]

xvecs = sc.parallelize([(raw[0], vec) for raw, vec in zip(x100, lowRank)])

# COMMAND ----------

largeValues = 0
maxValue, minValue = simMat[0][0], simMat[0][0]
for i in range(REMAIN):
  for j in range(REMAIN):
    if j > i:
      maxValue = max(maxValue, simMat[i][j])
      minValue = min(minValue, simMat[i][j])
    if j > i and simMat[i][j] >= 0.00009: #0.3:
      largeValues = largeValues + 1
largeValues, minValue, maxValue

# COMMAND ----------

pairs.map(lambda (x, y): y).distinct().count()

# COMMAND ----------

from matplotlib import pyplot as plt

TIMES = 1.0 #1000.0
plotMat = np.copy(simMat) * TIMES
maxPoint = TIMES * sorted([simMat[i][j] for i in range(len(simMat)) for j in range(i + 1, len(simMat))], reverse = True)[3]
plotMat[plotMat >= maxPoint] = maxPoint

fig, ax = plt.subplots(2)
heatmap = []
heatmap.append(ax[0].pcolor(plotMat))


(xcluster, xassign), (ycluster, yassign) = getPartition(4, 5, runs = 15, maxIterations = 10000, SEED = seeds[1])
xgroups = xassign.map(lambda (x, g): (g[0], x)) \
                 .groupByKey() \
                 .map(lambda (gid, g): g) \
                 .sortBy(lambda g: len(g)) \
                 .collect()

newMap, newMatIdx = {}, 0
newMat = [[-1.0 for i in range(REMAIN)] for j in range(REMAIN)]
for xg in xgroups:
  for x in xg:
    newMap[reverseX100[x]] = newMatIdx
    newMatIdx = newMatIdx + 1
for a in range(REMAIN):
  for b in range(REMAIN):
    newMat[newMap[a]][newMap[b]] = plotMat[a][b]

heatmap.append(ax[1].pcolor(np.array(newMat)))

cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = fig.colorbar(heatmap[0], cax=cax, ticks= np.arange(0.0, 1.0, 0.1), format='%.1f')
display(fig)

# COMMAND ----------

for xg in xgroups:
  print len(xg)
  print ', '.join(xg)
  print

# COMMAND ----------

for xg in xgroups:
  print len(xg)
  print ', '.join(xg)
  print

# COMMAND ----------

allitemfreq = []
for i in range(REMAIN):
  for j in range(i + 1, REMAIN):
    x = simMat[i][j]
    allitemfreq.append(x)
fig, ax = plt.subplots(1)
histogram = ax.hist(allitemfreq, log = True, bins = 100)
display(fig)

# COMMAND ----------

# MAGIC %md ## Minimum Description Length

# COMMAND ----------

from scipy.stats import entropy
entropy2 = lambda x: entropy(x, base = 2.0)

def getCodeLength(xassign, yassign):
  '''
    L(M, S) \propto
        log{m} + (m/2) * log{ |X| } + |X| H(P_x)
      + log{n} + (n/2) * log{ |Y| } + |Y| H(P_y)
      + (m * n)/2 * log{# of pairs}
      + \sum_{C_x, C_y} {
        L = # of pairs
        N_xy = # of pairs in the overlap of C_x and C_y
        N_x = # of pairs in C_x
        N_y = # of pairs in C_y
        return 1.0 / ( (L * N_xy) / (N_x * N_y) )
      }
  '''
  xMap = getKmeansMap(xassign)
  yMap = getKmeansMap(yassign)
  groupPairs = pairs.map(lambda (x, y): ((xMap[x], yMap[y]), 1)) \
                    .reduceByKey(add)
  S = groupPairs.map(lambda (p, c): c).sum()
  Tx = groupPairs.map(lambda ((x, y), c): (x, c)) \
                 .reduceByKey(add) \
                 .sortByKey() \
                 .map(lambda (x, c): c) \
                 .collect()
  Ty = groupPairs.map(lambda ((x, y), c): (y, c)) \
                 .reduceByKey(add) \
                 .sortByKey() \
                 .map(lambda (y, c): c) \
                 .collect()
  Tx = 1.0 * np.array(Tx) / S
  Ty = 1.0 * np.array(Ty) / S
  m, n = len(set(xMap.values())), len(set(yMap.values()))
  Nx, Ny = xassign.count(), yassign.count()

  model = log2(m) + (m / 2.0) * log2(Nx) + Nx * max(1.0, entropy2(Tx)) \
            + log2(n) + (n / 2.0) * log2(Ny) + Ny * max(1.0, entropy2(Ty)) \
            + (m * n / 2.0) * log2(S)
  data = groupPairs.map(lambda ((x, y), c): (c, (1.0 * c / S) / (Tx[x] * Ty[y]))) \
                   .map(lambda (c, p): c * log2(1.0 / p)) \
                   .sum()
  return model, data, model + data

# COMMAND ----------

def partitionProb(xassign, yassign):
  xMap = getKmeansMap(xassign)
  yMap = getKmeansMap(yassign)
  m, n = len(set(xMap.values())), len(set(yMap.values()))
  groupPairs = pairs.map(lambda (x, y): ((xMap[x], yMap[y]), 1)) \
                    .reduceByKey(add)

  S = groupPairs.map(lambda (p, c): c).sum()
  Tx = groupPairs.map(lambda ((x, y), c): (x, c)) \
                 .reduceByKey(add) \
                 .sortByKey() \
                 .map(lambda (x, c): c) \
                 .collect()
  Tx = 1.0 * np.array(Tx) / S
  Ty = groupPairs.map(lambda ((x, y), c): (y, c)) \
                 .reduceByKey(add) \
                 .sortByKey() \
                 .map(lambda (y, c): c) \
                 .collect()
  Ty = 1.0 * np.array(Ty) / S
  Txy = [[0.0 for j in range(n)] for i in range(m)]
  for (x, y), p in groupPairs.map(lambda (key, c): (key, 1.0 * c / S)).collect():
    Txy[x][y] = p
  return Tx, Ty, Txy

# COMMAND ----------

xgroup = pairs.map(lambda (x, y): x) \
               .distinct() \
               .map(lambda x: (x, (0, 0.0, 1000.0)))
ygroup = pairs.map(lambda (x, y): y) \
               .distinct() \
               .map(lambda y: (y, (0, 0.0, 1000.0)))
print getCodeLength(xgroup, ygroup)
mdl = {}

# COMMAND ----------

numOfCentersX = 4
for numOfCentersY in range(20, 21):
  (xcluster, xassign), (ycluster, yassign) = getPartition(numOfCentersX, numOfCentersY, runs = 15, maxIterations = 10000, SEED = seeds[1])
  mdl[(numOfCentersX, numOfCentersY)] = getCodeLength(xassign, yassign)

# COMMAND ----------

mdl

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(list(range(1,21)),map(lambda p: p[1][-1], sorted(mdl.items())))
ax.set_xlim(1, 20)
display(fig)

# COMMAND ----------

# old 1
mdl

# COMMAND ----------

# old
mdl

# COMMAND ----------

# MAGIC %md ## Just look at (4, 5)

# COMMAND ----------

(xcluster, xassign), (ycluster, yassign) = getPartition(4, 5, runs = 15, maxIterations = 10000, SEED = seeds[1])
ygroups = getKmeansGroup(yassign)

# COMMAND ----------

for yg in ygroups:
  print len(yg),
print

# COMMAND ----------

Tx, Ty, Txy = partitionProb(xassign, yassign)
print "Rows:\t",
for p in Tx:
  print '%.2lf' % p,
print '\n'
print "Columns",
for p in Ty:
  print '%.2lf' % p,
print '\n'
for i in range(len(Txy)):
  for j in range(len(Txy[0])):
    print "%.4f(%.2f)" % (Txy[i][j], Txy[i][j] / (Tx[i] * Ty[j])), '\t',
  print

# COMMAND ----------

print

# COMMAND ----------

for center in ycluster.centers:
  print center

# COMMAND ----------

Tx, Ty, Txy = partitionProb(xassign, yassign)
print "Rows:\t",
for p in Tx:
  print '%.2lf' % p,
print '\n'
print "Columns",
for p in Ty:
  print '%.2lf' % p,
print '\n'
for i in range(len(Txy)):
  for j in range(len(Txy[0])):
    print "%.4f(%.2f)" % (Txy[i][j], Txy[i][j] / (Tx[i] * Ty[j])), '\t',
  print

# COMMAND ----------

# MAGIC %md ## Stability

# COMMAND ----------

def getAssignmentEntropy(assign):
  count = assign.map(lambda (a, s): (s, 1)) \
                .reduceByKey(add) \
                .map(lambda (s, c): c) \
                .collect()
  count = np.array(count)
  return entropy2(1.0 * count / count.sum())

def merge(assign, groups):
  def isin(item, group):
    first = -1
    last = len(group)
    while first + 1 < last:
      midpoint = (first + last) / 2
      if group[midpoint] > item:
        last = midpoint
      elif group[midpoint] < item:
        first = midpoint
      else:
        return True
    return False

  def getId(item):
    for gid, g in enumerate(groups):
      if isin(item, g):
        return gid
  return assign.map(lambda (a, s): (a, s + ',%d' % getId(a)))

# COMMAND ----------

# MAGIC %md ### runs = 3

# COMMAND ----------

(xcluster, xassign), (ycluster, yassign) = getPartition(4, 4, runs = 15, maxIterations = 10000, SEED = seeds[1])
ygroups = [sorted(yg) for yg in getKmeansGroup(yassign)]
Tx, Ty, Txy = partitionProb(xassign, yassign)
for i in range(len(ygroups)):
  print "Group %d" % (i + 1)
  print "Size = %d" % len(ygroups[i])
  print "feature:", ycluster.centers[i]
  for j in range(len(ycluster.centers[i])):
    print "%.2lf(%.2lf)" % (Txy[i][j], Txy[i][j] / (Tx[i] * Ty[j])),
  print '\n'

# COMMAND ----------

xsig03 = pairs.map(lambda (x, y): (x, ''))
xentropyHist03 = []
ysig03 = pairs.map(lambda (x, y): (y, ''))
entropyHist03 = []

# COMMAND ----------

seeds = [893424, 781410] \
        + [998018, 810364, 356054, 558933, 275442, 495409, 632316, 78738] \
        + [798831, 133566, 104403, 548919, 857559, 676288, 947344, 305413, 571594, 822052]
for rseed in seeds:
  (xcluster, xassign), (ycluster, yassign) = getPartition(4, 4, runs = 15, maxIterations = 10000, SEED = rseed)
  ygroups = [sorted(yg) for yg in getKmeansGroup(yassign)]
  ysig03 = merge(ysig03, ygroups)
  entropyHist03.append(getAssignmentEntropy(ysig03))
  xgroups = [sorted(xg) for xg in getKmeansGroup(xassign)]
  xsig03 = merge(xsig03, xgroups)
  xentropyHist03.append(getAssignmentEntropy(xsig03))

# COMMAND ----------

print xentropyHist03
print entropyHist03

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(range(1, 21), xentropyHist03)
ax.set_title("Entropy of the Hashtags Partition with Multiple Initial Sets")
ax.set_xlim(left = 1)
display(fig)

# COMMAND ----------

for p in ygroups:
  print len(p)

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(range(1, 21), entropyHist03)
ax.set_title("Entropy of the Users Partition with Multiple Initial Sets")
ax.set_xlim(left = 1)
display(fig)

# COMMAND ----------

entropyHist03

# COMMAND ----------

print ysig03.map(lambda (y, sig): sig[:11] + sig[13:-1]).distinct().count()
print getAssignmentEntropy(ysig03.mapValues(lambda sig: sig[:11] + sig[13:-1]))

# COMMAND ----------

print ysig03.map(lambda (y, sig): sig).distinct().count()
print getAssignmentEntropy(ysig03)

# COMMAND ----------

# MAGIC %md ### old

# COMMAND ----------

print xentropyHist0
print entropyHist0

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(range(1, 21), xentropyHist0)
ax.set_xlim(left = 1, right = 20)
ax.set_title("Entropy of the Hashtags Partition")
display(fig)

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(range(1, 21), entropyHist0)
ax.set_xlim(left = 1, right = 20)
ax.set_title("Entropy of the Users Partition")
display(fig)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Assign only if close enough

# COMMAND ----------

seeds = [893424, 781410] \
        + [998018, 810364, 356054, 558933, 275442, 495409, 632316, 78738] \
        + [798831, 133566, 104403, 548919, 857559, 676288, 947344, 305413, 571594, 822052]

# COMMAND ----------

# Average
for rseed in seeds[:2]:
  (xcluster, xgroups, xerrors), (ycluster, ygroups, yerrors) = getPartition(4, 4, SEED = rseed)
  avg = yerrors.sum() / yerrors.count()
  var = yerrors.map(lambda e: (e - avg)**2).sum() / yerrors.count()
  print avg, var

# COMMAND ----------

# Variance
for rseed in seeds[:10]:
  diffs = distDiff(4, 4, SEED = rseed)
  avg = diffs.sum() / diffs.count()
  var = diffs.map(lambda e: (e - avg)**2).sum() / diffs.count()
  print avg, var


# COMMAND ----------

ysig = pairs.map(lambda (x, y): (y, ''))
entropyHist = []

# COMMAND ----------

#for rseed in seeds[:1]:
for rseed in seeds[1:]:
  (xcluster, xassign), (ycluster, yassign) = \
     getPartition(4, 4, diffFunc = lambda a, b: 1.0 * b / a, SEED = rseed)
  ygroups = [sorted(yg) for yg in getKmeansGroup(yassign, func = lambda ratio: ratio >= 2.0)]
  ysig = merge(ysig, ygroups)
  entropyHist.append(getAssignmentEntropy(ysig))

# COMMAND ----------

#entropyHist = []
for i in range(10, 20):
  count = ysig.map(lambda (a, s): (s[:(2*i+2)], 1)) \
              .reduceByKey(add) \
              .map(lambda (s, c): c) \
              .collect()
  count = np.array(count)
  entropyHist.append(entropy2(1.0 * count / count.sum()))
print entropyHist

# COMMAND ----------

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(entropyHist)
display(fig)

# COMMAND ----------

temp = getKmeansGroup(yassign, func = lambda ratio: ratio >= 2.0)
tempSum = np.sum([len(yg) for gid, yg in temp])
for gid, yg in temp:
  print gid, len(yg), 1.0 * len(yg) / tempSum

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Combine items that misclassified just once

# COMMAND ----------

def extendSig(sig, update):
  mapped = [False for i in range(len(sig))]
  taken = [False for i in range(len(sig))]
  done = 0
  last = {sig[i][-1]: i for i in range(len(sig))}
  for pair in update:
    if done >= len(sig):
      break
    items = pair.split(',')
    lst, cur = last[items[0]], items[1]
    if not mapped[lst] and not taken[int(cur)]:
      mapped[lst] = taken[int(cur)] = True
      sig[lst] = sig[lst] + (',%s' % cur)

def matchSig(target):
  r, cr = None, 0
  for sig in mainSig:
    diff = np.sum([1 for (s1, s2) in zip(sig, target) if s1 != s2])
    if diff == 0:
      return target
    if diff == 1:
      r = sig
      cr = cr + 1
  if cr == 1:
    return r
  return target

# COMMAND ----------

relaxEntropy = [entropyHist0[0]]

# First digit
mainSig = ysig0.map(lambda (a, s): s[:2]) \
               .distinct() \
               .collect()

# COMMAND ----------

for i in range(2, 20):
  last, cur = 2 * i - 1, 2 * i + 1
  update = ysig0.map(lambda (a, s): (s[last:cur + 1], 1)) \
                .reduceByKey(add) \
                .sortBy(lambda (sig, c): c, ascending = False) \
                .map(lambda (s, c): s) \
                .collect()
  extendSig(mainSig, update)

  count = ysig0.map(lambda (a, s): (matchSig(s[:(cur + 1)]), 1)) \
               .reduceByKey(add) \
               .map(lambda (s, c): c) \
               .collect()
  count = np.array(count)
  relaxEntropy.append(entropy2(1.0 * count / count.sum()))
print relaxEntropy

# COMMAND ----------

mainSig, relaxEntropy

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Playground 1: Number of different bits to some other user

# COMMAND ----------

def countdiff(a, b):
  count = 0
  for i in range(len(a)):
    if a[i] != b[i]:
      count = count + 1
  return count

allsig = sorted(yassign.map(lambda (key, s): s).distinct().collect())
diffcount = {}
for s in allsig:
  least = 999999
  for spr in allsig:
    if spr != s and least > countdiff(spr, s):
      least = countdiff(spr, s)
  diffcount[least] = diffcount.get(least, 0) + 1

print diffcount
del allsig

# COMMAND ----------

yassign.map(lambda (key, s): s).distinct().count()

# COMMAND ----------

yassign.map(lambda (key, s): (s, 1)).reduceByKey(add).sortBy(lambda p: p[1], ascending = False).take(20)

# COMMAND ----------

import gc
gc.collect()

# COMMAND ----------

