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


# get the dataset
lines = read_lines("dataset/movie_lines.txt")
conversations = read_lines("dataset/movie_conversations.txt")

