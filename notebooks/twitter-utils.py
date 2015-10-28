# Databricks notebook source exported at Wed, 28 Oct 2015 07:25:33 UTC

import json
import re
from collections import Counter
from operator import add

def tryLoad(l):
    try:
        return json.loads(l)
    except:
        return None


def extractTags(text):
    tags = re.findall("#\w[\w']*", text)
    return [tag.lower() for tag in tags]


def rawToJson(dataURL):
    return sc.textFile(dataURL) \
             .map(tryLoad) \
             .filter(lambda x: x)


def getAllTags(data):
    return data.map(lambda p: set(extractTags(p['text']))) \
               .reduce(lambda a, b: a.union(b))


def getAllUsers(data):
    return data.map(lambda p: {p['user']['id']}) \
               .reduce(lambda a, b: a.union(b))


def getAllDates(data):
    return data.filter(lambda p: 'created_at' in p) \
               .map(lambda p: \
                    strptime(p['created_at'], "%a %b %d %H:%M:%S +0000 %Y"))


def getUserTagsRDD(data, tagsFilter):
    tags = list(getAllTags(data))
    if tagsFilter:
        tags = [tag for tag in tags if tag in tagsFilter]
    tagsId = sc.broadcast({b: a for a, b in enumerate(tags)})
    usertags = data.map(lambda p: (p["user"]["id"], extractTags(p["text"]))) \
                   .mapValues(lambda tags: \
                        [tag for tag in tags if tag in tagsId.value])
    tagsCount = usertags.mapValues(lambda tags: \
                            Counter({tagsId.value[t]: 1 for t in tags})) \
                        .reduceByKey(add)
    return dict(enumerate(tags)), tagsCount


def getTagUsersRDD(data, usersFilter):
    users = list(getAllUsers(data))
    if usersFilter:
        users = [user for user in users if user in usersFilter]
    usersId = sc.broadcast({b: a for a, b in enumerate(users)})
    tagusers = data.map(lambda p: (p["user"]["id"], extractTags(p["text"]))) \
                   .filter(lambda (u, tags): u in usersId.value) \
                   .flatMap(lambda (u, tags): [(t, u) for t in tags])
    usersCount = tagusers.mapValues(lambda user: Counter({user: 1})) \
                         .reduceByKey(add)
    return dict(enumerate(users)), usersCount
