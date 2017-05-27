from builtins import print
from rake2 import *
from os import path,system
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from tf_idf import *
import codecs
from zad1 import lemmatize

def show_chart(keywordsTable, max_elements, title):
    x_values = []
    y_values = []
    x_ticks = []

    sortedKeywordsTable = sorted(keywordsTable, key=lambda elem: elem[1], reverse=True)
    totalKeywords = len(sortedKeywordsTable)

    for i in range(max_elements):
        if i >= totalKeywords:
            break
        x_values.append(i)
        y_values.append(sortedKeywordsTable[i][1])
        x_ticks.append(sortedKeywordsTable[i][0])

    plt.figure()
    plt.xticks(x_values, x_ticks, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.bar(x_values, y_values, 0.8)
    plt.savefig('wykres.png')
    plt.show()

def run_rake(text):
    # Split text into sentences
    sentenceList = split_sentences(text)
    stoppath = "SmartStoplist.txt"
    # stoppath = "stopwords-pl.txt"
    stopwordpattern = build_stop_word_regex(stoppath)


    #### RAKE ####
    # generate candidate keywords
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
    # calculate individual word scores
    wordscores = calculate_word_scores(phraseList)
    # generate candidate keyword scores
    keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)

    rake = Rake("SmartStoplist.txt",5, 3, 4)
    # rake = Rake("stopwords-pl.txt")
    keywords = rake.run(text)

    print(keywords)
    show_chart(keywords, 20, 'Rake')

    # a=lemmatize(text)
    # sentenceList = split_sentences(a)
    # phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
    # wordscores = calculate_word_scores(phraseList)
    # keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
    # sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)
    # keywords2 = rake.run(text)

    frequencies={}
    for keyword in keywords:
        # print(keyword)
        str = keyword[0]
        val= keyword[1]
        frequencies[str]=keyword[1]
    #### RAKE ####

    # Generate a word cloud image
    wordcloud = WordCloud(max_font_size=30,background_color='white', max_words=50).generate_from_frequencies(frequencies)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Rake')
    plt.axis("off")
    plt.show()

    # for keyword in keywords2:
    #     # print(keyword)
    #     str = keyword[0]
    #     val= keyword[1]
    #     frequencies[str]=keyword[1]
    # print(frequencies)

    # wordcloud = WordCloud(max_font_size=30,background_color='white').generate_from_frequencies(frequencies)
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")

def run_tf(text):
    # TF
    print("TF")
    stop=load_stop_words("SmartStoplist.txt")
    text=lemmatize(text, stop)
    freq = analize_tf(text)
    sortedFreq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedFreq)
    # print(freq)
    wordcloud = WordCloud(max_font_size=30,background_color='white', max_words=50).generate_from_frequencies(freq)

    word_tab=[]
    for str, fr in wordcloud.words_.items():
        word_tab.append([str, fr])

    show_chart(word_tab, 20, 'TF')

    plt.figure()
    plt.title('TF')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def run_tf_idf(text, texts):
    # TF-IDF
    print("TF-IDF")
    stop = load_stop_words("SmartStoplist.txt")

    text = lemmatize(text, stop)
    bloblist=[]
    for t in texts:
        bloblist.append(lemmatize(t, stop))
    freq = analize_tfidf(text, bloblist)
    print(freq)
    wordcloud = WordCloud(max_font_size=30,background_color='white', max_words=50).generate_from_frequencies(freq)

    word_tab = []
    for str, fr in wordcloud.words_.items():
        word_tab.append([str, fr])

    show_chart(word_tab, 20, 'TF-IDF')

    plt.figure()
    plt.title('TF-IDF')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def run_lda(text):
    print("LDA Algorithm")
    os.system("lda.py 1")

def run_lsa(text):
    print("LSA Algorithm")
    os.system("runClassification_LSA.py 1")
    os.system("inspect_LSA.py 1")


d = path.dirname(__file__)
text = open(path.join(d, 'lotr.txt'), encoding='utf8').read()
text2 = open(path.join(d, 'tekst.txt'), encoding='utf8').read()
# text_temp = codecs.open(path.join(d, 'tekst2.txt'), encoding='utf8')
# text = text_temp.readlines()[0]
# print(text)

run_rake(text)
run_tf(text)
run_tf_idf(text,[text,])
# run_lda(text)
# run_lsa(text)