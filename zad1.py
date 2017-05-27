from os import path
import sys

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

speech_part={'ADJ':'a', 'ADV':'r', 'NOUN':'n', 'VERB':'v'}

def lemmatize(text, stop):
    lemmatized_text = ""

    tokenized_text = nltk.word_tokenize(text);
    pos_tag_text = nltk.pos_tag(tokenized_text, tagset='universal')

    wordnet_lemmatizer = WordNetLemmatizer()

    for word in pos_tag_text:
        #print(wordnet_lemmatizer.lemmatize(word[0], pos=speech_part.get(word[1], 'n')))
        if word[0].lower() in stop:
            continue
        lemmatized_text+=wordnet_lemmatizer.lemmatize(word[0].lower(), pos=speech_part.get(word[1], 'n'))+" "

    return lemmatized_text

# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))
#
# d = path.dirname(__file__)
# text = open(path.join(d, 'tekst.txt')).read()
#
# print(lemmatize(text))

# stemmer = SnowballStemmer("english")
# print(stemmer.stem("generalized"))
# print(stemmer.stem("generalization"))

#text = nltk.word_tokenize("are")
#print(nltk.pos_tag(text,tagset='universal'))
# print(nltk.pos_tag(nltk.word_tokenize("dog"),tagset='universal'))
# print(nltk.pos_tag(nltk.word_tokenize("new"),tagset='universal'))
# print(nltk.pos_tag(nltk.word_tokenize("on"),tagset='universal'))
# print(nltk.pos_tag(nltk.word_tokenize("still"),tagset='universal'))

# tokenized_text=nltk.word_tokenize(text);
# pos_tag_text=nltk.pos_tag(tokenized_text,tagset='universal')
#
# wordnet_lemmatizer = WordNetLemmatizer()
# for a in pos_tag_text:
#     print(a," ",speech_part.get(a[1],'n')," " ,wordnet_lemmatizer.lemmatize(a[0], pos=speech_part.get(a[1],'n')))
#
# print(wordnet_lemmatizer.lemmatize("dogs"))
# print(wordnet_lemmatizer.lemmatize("are"))
# print(wordnet_lemmatizer.lemmatize("are",pos='VBP'))
