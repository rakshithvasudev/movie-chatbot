# import the necessary libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from collections import Counter
import time


# --------------------- DATA PREPROCESSING ----------------------------------
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
# single quotes, commas & whitespace.
for conversation in conversations[:-1]:
    _conversation = conversation.split(delimiter)[-1][1:-1]. \
        replace("'", "").replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))

# generate question and answers - the original dataset is structured
# so that for every first id, the corresponding answer is present in the next id.
# ex: [Q: L194, A: L195, Q: L196, A: L197], therefore question & answer list can be
# built as follows.
questions = []
answers = []

for conversation in conversations_ids:
    for idx in range(len(conversation) - 1):
        questions.append(id2line[conversation[idx]])
        answers.append(id2line[conversation[idx + 1]])


def clean_text(text):
    """
    Removes special characters, gets the correct version
    of the text.
    :param text: string text source to be cleaned.
    :return: cleaned string text.
    """
    # convert text to lowercase
    text = text.lower()

    # replace every occurrence of "i'm" with "i am"
    text = re.sub("i'm", "i am", text)

    # replace every occurrence of "he's" with "he is"
    text = re.sub("he's", "he is", text)

    # replace every occurrence of "she's" with "she is"
    text = re.sub("she's", "she is", text)

    # replace every occurrence of "whats's" with "what is"
    text = re.sub("what's", "what is", text)

    # replace every occurrence of "where's" with "where is"
    text = re.sub("where's", "where is", text)

    # replace every occurrence of "'ll" with "will"
    text = re.sub("\'ll", " will", text)

    # replace every occurrence of "'ve" with "have"
    text = re.sub("\'ve", " have", text)

    # replace every occurrence of "'s" with " is"
    text = re.sub("\'s", " is", text)

    # replace every occurrence of "'ve" with "have"
    text = re.sub("\'re", " are", text)

    # replace every occurrence of "'d" with "would"
    text = re.sub("\'d", " would", text)

    # replace every occurrence of "won't" with "will not"
    text = re.sub("won't", "will not", text)

    # replace every occurrence of "can't" with "cannot"
    text = re.sub("can't", "cannot", text)

    # replace special characters with "" - remove them
    text = re.sub("[-()\"#/@;:<>{}+=|.?,^%]", "", text)

    return text


def clean_string_list(string_list):
    """
    cleans every element of the string_list text & returns a clean list.
    :param string_list: source list that needs to be cleaned.
    :return: cleaned elements of the list.
    """
    cleaned_string = []
    for string_element in string_list:
        cleaned_string.append(clean_text(string_element))

    return cleaned_string


# clean the questions and answer elements
clean_questions = clean_string_list(questions)
clean_answers = clean_string_list(answers)


def count_words(string_list):
    """
    Counts the number of words in a given list of sentences.
    :param string_list: list of sentences.
    :return: word counts in the entire list of corpus.
    """
    # keep a track of word counts
    word_counts = {}

    # iterate through every sentence in string_list
    for sentence in string_list:
        # capture every word - using " " as delimiter
        for word in sentence.split():
            # populate word_counts accordingly
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    return word_counts


# get the word counts for questions and answers
word2count_question = count_words(clean_questions)
word2count_answers = count_words(clean_answers)

# capture the total count of word frequency in word2count as a dictionary
word2count = dict(Counter(word2count_question) + Counter(word2count_answers))

# the least value of word_counts below which the word itself gets
# discarded.
threshold = 20


def words2int(dictionary, threshold):
    """
    Takes the words from the dictionary filters them based on
    the dictionary word count, if their count exceeds the threshold.
    The value counts are their corresponding words.
    :param dictionary: it contains the words and their corresponding count.
    :param threshold: filter parameter.
    :return: retained words and their unique counts.
    """
    selected_words = {}
    counter = 0
    for word, count in dictionary.items():
        if count >= threshold:
            selected_words[word] = counter
            counter += 1
    return selected_words


questionwords2int = words2int(word2count, 20)
answerswords2int = words2int(word2count, 20)

# add last tokens to dictionaries - order important
# <PAD> - padding the content
# <EOS> - End of string
# <OUT> - Filtered out
# <SOS> - Start of string
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

# this is needed to deal with seq2seq model.

# add the tokens for questions
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1

# add the tokens for answers
for token in tokens:
    answerswords2int[token] = len(questionwords2int) + 1

# create inverse dictionary of answerwords2int dictionary
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

# add the EOS token- this is needed at the end of the decoding layers for
# seq2seq model.
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

# turning questions and answers to integers
# & replacing filtered out words by int <OUT> token

# all the questions converted to int
questions_into_int = []
for question in clean_questions:
    # integers for the corresponding question
    question_int = []
    # iterate through every word in the question
    for word in question.split():
        if word not in questionwords2int:
            question_int.append(questionwords2int["<OUT>"])
        else:
            question_int.append(questionwords2int[word])
    questions_into_int.append(question_int)

# all the answers converted to int
answers_into_int = []
for answer in clean_answers:
    # integers for the corresponding answer
    answer_int = []
    # iterate through every word in the answer
    for word in answer.split():
        if word not in answerswords2int:
            answer_int.append(answerswords2int["<OUT>"])
        else:
            answer_int.append(answerswords2int[word])
        answers_into_int.append(answer_int)

# sorting questions and answers by length: this is necessary
# because it improves(speeds up) the training process, converges faster
# in that it'll reduce the padding during the training.
sorted_clean_questions = []
sorted_clean_answers = []

# It is important to include questions that are not too long.
# It is better to start from 1 include up to 20 or 25(arbitrary) words.
# This is to makes the training efficient.
# +1 for ensuring the count goes upto 25 - range upperbound.
# this could be run in O(nlog(n)) as opposed to o(n), if sort function
# is used.
for length in range(1, 25 + 1):
    # get the index and actual question words
    for i in enumerate(questions_into_int):
        # if the length of the question is the actual length in range
        if len(i[1]) == length:
            # append the actual question that was considered & the corresponding answer
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


# -------------------------Seq2Seq model building-----------------------------

def model_inputs():
    """
    Initializes the input for the nn model.
    :return: inputs, targets, learning rate, keep probability (Regularization)
    """
    inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target')
    lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob
