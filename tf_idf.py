import math
from textblob import TextBlob as tb
from os import path

def tf(word, blob, total_len):
    return blob.words.count(word) / total_len

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob,len(blob)) * idf(word, bloblist)

def analize_tf(text):
    blob = tb(text)
    total_len=len(tb(text))

    scores = {word: tf(word, blob, total_len) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    word_list = []
    for word, score in sorted_words:
        # print("Word: {}, TF: {}".format(word, round(score, 5)))
        word_list.append(word)
        #word_list.append(' ')
    #word_str = ''.join(word_list)
    return scores

def analize_tfidf(text, bloblist):
    blob = tb(text)
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1])
    print(scores)
    word_list = []
    for word, score in sorted_words: #[:3]:
        # print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
        word_list.append(word)
        #word_list.append(' ')
    #word_str = ''.join(word_list)
    print(sorted_words)
    return scores