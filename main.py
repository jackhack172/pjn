from builtins import print
from rake import *
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the whole text.
from zad1 import lemmatize

d = path.dirname(__file__)
text = open(path.join(d, 'tekst.txt')).read()
# text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."

# Split text into sentences
sentenceList = split_sentences(text)
#stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1
stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
stopwordpattern = build_stop_word_regex(stoppath)

# generate candidate keywords
phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)

# calculate individual word scores
wordscores = calculate_word_scores(phraseList)

# generate candidate keyword scores
keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
# if debug: print (keywordcandidates)

sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)
# if debug: print (sortedKeywords)

totalKeywords = len(sortedKeywords)
# if debug: print (totalKeywords)
# print (sortedKeywords[0:int(totalKeywords / 3)])

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
    print(str)
    frequencies[str]=keyword[1]
print(frequencies)

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(frequencies)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

for keyword in keywords2:
    # print(keyword)
    str = keyword[0]
    val= keyword[1]
    print(str)
    frequencies[str]=keyword[1]
print(frequencies)

wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(frequencies)
# lower max_font_size
# wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
plt.show()


print("LDA Algorithm")
os.system("lda.py 1")

print("LSA Algorithm")
os.system("runClassification_LSA 1")
os.system("inspect_LSA 1")