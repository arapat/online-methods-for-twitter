# Databricks notebook source exported at Wed, 16 Dec 2015 17:21:28 UTC
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
dataStDate = 10
dataStep = 20
dataSt, dataEd = dataStDate, dataStDate + dataStep

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

# Construct estimation vectors
hashFunc = HashMD5(hashLength)
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

print 'Number of users:', numOfUsers

# COMMAND ----------

# MAGIC %md # Algorithm

# COMMAND ----------

# MAGIC %md ## Find initial pairs

# COMMAND ----------

def getBestPairs(xInfo, gid, beta):
  '''
  x must occur at least maxCount * beta times to be considered
  '''
  data = xInfo.filter(lambda (x, (_gid, est)): _gid == gid) \
              .map(lambda (x, (gid, est)): (x, est))
  maxCount = data.map(lambda (x, est): getUniqueItemsApproximate(est[1], numGroups, groupSize)) \
                 .reduce(max)
  x = data.filter(lambda (x, est): getUniqueItemsApproximate(est[1], numGroups, groupSize) >= maxCount * beta) \
          .collect()
  
  length = len(x)
  pairs = []
  for i in range(length):
    name1, est1 = x[i]
    count1 = getUniqueItemsApproximate(est1[1], numGroups, groupSize)
    for j in range(i + 1, length):
      name2, est2 = x[j]
      count2 = getUniqueItemsApproximate(est2[1], numGroups, groupSize)
      p = getSimilarity(est1[0], est2[0])
      pairs.append((name1, name2, count1, count2, p))
  return pairs

# COMMAND ----------

beta = 0.5
scoreFunc = lambda (n1, n2, c1, c2, p): p - beta * (c1 + c2) / (1 + p) / numOfUsers 

# pairs = getBestPairs(xInfo, 0, 0.1)
for p in sorted(pairs, key = scoreFunc)[:10]:
  s = scoreFunc(p)
  n1, n2, c1, c2, pp = p
  s1, s2 = pp, (c1 + c2) / (1 + pp) / numOfUsers
  print p[0], p[1], s, s1, s2

# COMMAND ----------

# MAGIC %md ## Grouping

# COMMAND ----------

# Functions for creating log table
logStamp = 0
tagsTable = []
setsTable = []

def writeLogs(updateDict):
  global logStamp
  global tagsTable
  global setsTable
  global xInfo

  # Item logs
  logStamp = logStamp + 1
  for x, values in updateDict.items():
    gid, score, _log = values
    count, background, bad_fit, good_fit = 'n/a', 'n/a', 'n/a', 'n/a'
    if _log:
      count, background, bad_fit, good_fit = _log
    tagsTable.append({
        'iter': logStamp,
        'name': x,
        'group': gid,
        'score': score,
        'count': count,
        'background': background,
        'bad-fit': bad_fit,
        'good-fit': good_fit
      })
  # Set logs
  setEsts = xInfo.map(lambda (x, (gid, est)): (gid, est)) \
                 .reduceByKey(reduceEstimates) \
                 .sortByKey() \
                 .map(lambda (gid, est): est) \
                 .collect()
  setSizes = xInfo.map(lambda (x, (gid, est)): (gid, 1)) \
                  .reduceByKey(add) \
                  .sortByKey() \
                  .map(lambda (gid, count): count) \
                  .collect()
  content = {'iter': logStamp}
  for idx, (est, size) in enumerate(zip(setEsts, setSizes)):
    content['#items %d' % idx] = size
    content['#features %d' % idx] = getUniqueItemsApproximate(est[1], numGroups, groupSize)
  for j in range(len(setEsts)):
    n1 = getUniqueItemsApproximate(setEsts[j][1], numGroups, groupSize)
    for k in range(j + 1, len(setEsts)):
      n2 = getUniqueItemsApproximate(setEsts[k][1], numGroups, groupSize)
      p = getSimilarity(setEsts[j][0], setEsts[k][0])
      both = p * (n1 + n2) / (1 + p)
      content['#features %d ^ %d' % (j, k)] = '%d (%f)' % (both, both / min(n1, n2))
  setsTable.append(content)

# COMMAND ----------

# Main algorithm
def clear():
  global xInfo
  global logStamp
  xInfo = xInfo.map(lambda (x, (gid, est)): (x, (0, est))) \
               .cache()
  logStamp = 0

def getDist(p, setEsts, newGroups, oldGroup):
  x, (_gid, est) = p
  upperRatio = getLogRatioDiff(est, setEsts[oldGroup], numOfUsers, numGroups, groupSize, delta, upper = True)
  ratios = [getLogRatioDiff(est, setEsts[gid], numOfUsers, numGroups, groupSize, delta) for gid in newGroups]
  ratio, argmax = np.max(ratios), np.argmax(ratios)
  gid = newGroups[argmax]

  # For logs: count, background, bad-fit, good-fit
  _logs = (getUniqueItemsApproximate(est[1], numGroups, groupSize), upperRatio, ratios[1 - argmax], ratio)
  return (x, (gid, ratio - upperRatio, _logs))

def createNewGroup(xsets, groups):
  global xInfo
  for gid, xset in zip(groups, xsets):
    update = lambda (x, (_gid, est)): (x, (gid, est)) if x in xset else (x, (_gid, est))
    xInfo = xInfo.map(update) \
                 .cache()
  # Write logs
  writeLogs({x: (gid, 'n/a', None) for xset, gid in zip(xsets, groups) for x in xset})

def runAlg(newGroups, oldGroup, threshold, fraction, minAdded, maxItr):
  global xInfo
  itr, newlyAdded = 0, np.maximum
  while newlyAdded >= minAdded and itr < maxItr:
    setEsts = xInfo.map(lambda (x, (gid, est)): (gid, est)) \
                   .reduceByKey(reduceEstimates) \
                   .sortByKey() \
                   .map(lambda (gid, est): est) \
                   .collect()
    qualified = xInfo.filter(lambda (x, (gid, est)): gid == oldGroup) \
                     .map(lambda p: getDist(p, setEsts, newGroups, oldGroup)) \
                     .filter(lambda (x, (gid, dist, _logs)): dist >= threshold) \
                     .cache()
    newlyAdded = int(qualified.count() * fraction)
    print newlyAdded
    updateDict = dict(qualified.take(int(qualified.count() * fraction)))
    update = lambda (x, (_gid, est)): (x, (updateDict[x][0], est)) if x in updateDict else (x, (_gid, est))
    xInfo = xInfo.map(update) \
                 .cache()

    writeLogs(updateDict)
    gc.collect()
    itr = itr + 1

# COMMAND ----------

clear()
isNew = True

# COMMAND ----------

# max delta for the bound
delta = 0.001

# minimum value for metric
threshold = np.log2(5) # 5, 3
# expand fraction
fraction = 0.1 # 0.1, 0.5
# Stop conditions
minAdded, maxItr = 1, 1

newGroups = [1, 2]
oldGroup = 0

xsets = [['standwithpp'], ['tcot']] # 0 => 1, 2

if isNew:
  createNewGroup(xsets, newGroups)
  isNew = False
runAlg(newGroups, oldGroup, threshold, fraction, minAdded, maxItr)

# COMMAND ----------

# MAGIC %md ## Print results

# COMMAND ----------

groups = xInfo.filter(lambda (x, (gid, est)): gid > 0) \
              .map(lambda (x, (gid, est)): (gid, x)) \
              .groupByKey() \
              .collect()
groups.sort()
for gid, items in groups:
  print gid, '(%d)' % len(items)
  print '====='
  print ', '.join(list(items)[:100])
  print

# COMMAND ----------

setEsts = xInfo.map(lambda (x, (gid, est)): (gid, est)) \
               .reduceByKey(reduceEstimates) \
               .sortByKey() \
               .map(lambda (gid, est): est) \
               .collect()
print 'All hashtags:', xInfo.count()
print 'All users:', numOfUsers
print 'Users in subsets (1, 2):', getUniqueItemsApproximate(setEsts[1][1], numGroups, groupSize), getUniqueItemsApproximate(setEsts[2][1], numGroups, groupSize)
print 'Users in background:', getUniqueItemsApproximate(setEsts[0][1], numGroups, groupSize)
print 'Intersections (0 & 1, 0 & 2, 1 & 2)', getSimilarity(setEsts[0][0], setEsts[1][0]), getSimilarity(setEsts[0][0], setEsts[2][0]), getSimilarity(setEsts[1][0], setEsts[2][0])

# COMMAND ----------

setEsts = xInfo.map(lambda (x, (gid, est)): (gid, est)) \
                   .reduceByKey(reduceEstimates) \
                   .sortByKey() \
                   .map(lambda (gid, est): est) \
                   .collect()
qualified = xInfo.filter(lambda (x, (gid, est)): gid == oldGroup) \
                 .map(lambda p: getDist(p, setEsts, newGroups, oldGroup)) \
                 .sortBy(lambda (x, (gid, dist)): dist, ascending = False) \
                 .take(100)
qualified

# COMMAND ----------

# MAGIC %md ## Print logs

# COMMAND ----------

def convert(x):
  if type(x) is np.float64:
    return x.item()
  return x

def createRows(content, attr):
  return [
    tuple([convert(p.get(k, 'n/a')) for k in attr]) for p in content
  ]

# COMMAND ----------

attr = ['iter', 'name', 'count', 'score', 'group', 'background', 'bad-fit', 'good-fit']
rows = sc.parallelize(createRows(tagsTable, attr))
df = sqlContext.createDataFrame(rows, attr)
display(df)

# COMMAND ----------

attr = sorted(setsTable[-1].keys(), key = lambda s: (len(s), s) if s != 'iter' else '')
rows = sc.parallelize(createRows(setsTable, attr))
df = sqlContext.createDataFrame(rows, attr)
display(df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Count number of users mentioned too few hashtags

# COMMAND ----------

xcounts = pairs.map(lambda (x, y): (x, 1)) \
               .reduceByKey(add) \
               .map(lambda (k, count): (count, 1)) \
               .reduceByKey(add) \
               .sortByKey() \
               .collect()

ycounts = pairs.map(lambda (x, y): (y, 1)) \
               .reduceByKey(add) \
               .map(lambda (k, count): (count, 1)) \
               .reduceByKey(add) \
               .sortByKey() \
               .collect()

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
ax[0].plot([a for a, b in xcounts], [b for a, b in xcounts])
ax[0].set_xlim(0, 100)
ax[1].plot([a for a, b in ycounts], [b for a, b in ycounts])
ax[1].set_xlim(0, 100)

vxcounts = [b for a, b in xcounts]
vycounts = [b for a, b in ycounts]
print sum(vxcounts), sum(vycounts)
print 1.0 * sum(vycounts[:2]) / sum(vycounts)
# display(fig)

# COMMAND ----------



# COMMAND ----------

