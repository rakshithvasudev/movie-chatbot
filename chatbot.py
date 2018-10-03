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


