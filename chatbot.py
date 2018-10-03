# import the necessary libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import time


def read_lines(source):
    """
    Read the file from the source location, split into lines, return
    the list of lines read.
    :param source: the location at which the file is present
    :return: list of lines
    """
    with open(source, encoding="ISO-8859-1", mode='r') as f:
        source_lines = f.read().split("\n")
    return source_lines


# get the dataset - source file:
# https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
# lines contains the the actual text of each utterance-
# 	- fields:
# 		- lineID
# 		- characterID (who uttered this phrase)
# 		- movieID
# 		- character name
# 		- text of the utterance
lines = read_lines("dataset/movie_lines.txt")

# conversations contains  movie_conversations.txt
# 	- the structure of the conversations
# 	- fields
# 		- characterID of the first character involved in the conversation
# 		- characterID of the second character involved in the conversation
# 		- movieID of the movie in which the conversation occurred
# 		- list of the utterances that make the conversation, in chronological
# 			order: ['lineID1','lineID2',�,'lineIDN']
# 			has to be matched with movie_lines.txt to reconstruct the actual content
conversations = read_lines("dataset/movie_conversations.txt")

# create a dictionary to keep a track of line_id and the actual line itself.
# split the text using the delimiter - '+++$+++', map the line id to the actual line.
# ex: 'L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!'
# here line_id is 'L1045', line is 'They do not!'
delimiter = " +++$+++ "
id2line = {}

# for every line split the line on the delimiter, then load the delimited
# tokens into the dictionary, where the id is the key, value is the actual line.
for line in lines:
    line_split = line.split(delimiter)
    if len(line_split) == 5:
        id2line[line_split[0]] = line_split[4]

# this list contains all the conversation line ids involved in a conversation
# using the conversations list.
conversations_ids = []

# for every conversation in conversations[:-1] (the last element is just '').
# get the list of lines_in the conversation, remove the square braces,
# single quotes & commas.
for conversation in conversations[:-1]:
    _conversation = conversation.split(delimiter)[-1][1:-1].replace("'", "").replace("'", "")
    conversations_ids.append(_conversation.split(","))


# question and answers