from builtins import print
from rake import *
from os import path,system
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from tf_idf import *
from zad1 import lemmatize

d = path.dirname(__file__)
text = open(path.join(d, 'tekst.txt')).read()

# Split text into sentences
sentenceList = split_sentences(text)
stoppath = "SmartStoplist.txt"
stopwordpattern = build_stop_word_regex(stoppath)

# generate candidate keywords
phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
# calculate individual word scores
wordscores = calculate_word_scores(phraseList)
# generate candidate keyword scores
keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)

sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)
totalKeywords = len(sortedKeywords)

rake = Rake("SmartStoplist.txt")
keywords = rake.run(text)

a=lemmatize(text)
sentenceList = split_sentences(a)
phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
wordscores = calculate_word_scores(phraseList)
keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)
keywords2 = rake.run(text)

frequencies={}
for keyword in keywords:
    # print(keyword)
    str = keyword[0]
    val= keyword[1]
    frequencies[str]=keyword[1]
print(frequencies)

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(frequencies)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

for keyword in keywords2:
    # print(keyword)
    str = keyword[0]
    val= keyword[1]
    frequencies[str]=keyword[1]
print(frequencies)

wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(frequencies)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# TF
print("TF")
freq = analize_tf(text)
print(freq)
wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(freq)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# TF-IDF
print("TF-IDF")
bloblist = [text]
freq = analize_tfidf(text, bloblist)
print(freq)
wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(freq)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()


print("LDA Algorithm")
os.system("lda.py 1")

print("LSA Algorithm")
os.system("runClassification_LSA.py 1")
os.system("inspect_LSA.py 1")