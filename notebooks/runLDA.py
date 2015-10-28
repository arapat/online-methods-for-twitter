# Databricks notebook source exported at Wed, 28 Oct 2015 17:08:05 UTC
# TODO: 
# 1. This code is not tested
# 2. Implement with new extractTags
files = [p.path for p in dbutils.fs.ls('/mnt/%s/data' % MOUNT_NAME)]
data = sc.textFile(','.join(files)).map(tryLoad).filter(lambda p: p and 'text' in p and 'user' in p).cache()
numOfTags = 500
topTags = data.flatMap(lambda p: extractTags(p['text'])) \
              .map(lambda w: (w, 1)) \
              .reduceByKey(add) \
              .sortBy(lambda p: p[1], ascending=False) \
              .map(lambda p: p[0]).take(numOfTags)
tagsDict, userTagsRDD = getUserTagsRDD(data, topTags)
print len(tagsDict)

ldaModel = runOnlineLDA(userTagsRDD, len(tagsDict))
topics = getTopics(ldaModel, tagsDict)
for tid, topic in enumerate(topics):
  print tid
  for word in topic[:20]:
    print word
  print