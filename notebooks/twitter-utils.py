# Databricks notebook source exported at Wed, 2 Mar 2016 23:01:51 UTC

import json
import re
from datetime import datetime
from collections import Counter
from operator import add

def _tryLoad(l):
    try:
        return json.loads(l)
    except:
        return None

def allTweetsFromFile(dataURL):
    return sc.textFile(dataURL) \
             .map(_tryLoad)

def tweetsFromFile(dataURL):
    return sc.textFile(dataURL) \
             .map(_tryLoad) \
             .filter(_hasHashtags)

def tweetsFromFileNoFilter(dataURL):
    return sc.textFile(dataURL) \
             .map(_tryLoad)

def _hasHashtags(jsonObj):
    return jsonObj and 'entities' in jsonObj and jsonObj['entities']['hashtags']


def getHashtags(jsonObj):
    return [p['text'].lower() for p in jsonObj['entities']['hashtags']]


def getDate(jsonObj):
    return datetime.strptime(jsonObj['created_at'], "%a %b %d %H:%M:%S +0000 %Y").date()
  

def getUser(jsonObj):
    return jsonObj['user']['id_str']


def getAllTags(data):
    return data.map(lambda p: set(getHashtags(p))) \
               .reduce(lambda a, b: a.union(b))


def getAllUsers(data):
    return data.map(lambda p: {p['user']['id_str']}) \
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
    usertags = data.map(lambda p: (getUser(p), getHashtags(p))) \
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
    tagusers = data.map(lambda p: (p["user"]["id_str"], getHashtags(p))) \
                   .filter(lambda (u, tags): u in usersId.value) \
                   .flatMap(lambda (u, tags): [(t, u) for t in tags])
    usersCount = tagusers.mapValues(lambda user: Counter({user: 1})) \
                         .reduceByKey(add)
    return dict(enumerate(users)), usersCount
