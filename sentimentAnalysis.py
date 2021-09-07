#nltk library import statments
from re import S
import nltk
import string
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download(["names", "stopwords", "vader_lexicon", "punkt"])

#Initializes the textReader
inputText = open('input.txt').read()
sentimentAnalyzer = SentimentIntensityAnalyzer()

#Makes an copy and keeps all the text in a single string with the proper punctuation and lowercase letters, calculates the sentiment score
cleanedFullText = inputText.translate(str.maketrans('', '', string.punctuation))
cleanedFullText = cleanedFullText.lower()
print("Sentiment Analysis for full text:", sentimentAnalyzer.polarity_scores(cleanedFullText)['compound'])#Prints out the calculated sentiment score

#Makes a copy of the text and tokenizes it into sentences with lowercase letters, proper punctuation, and no "\n" characters
sentenceText = nltk.sent_tokenize(inputText)
sentenceText = [x.translate(str.maketrans('', '', string.punctuation)) for x in sentenceText]
sentenceText = [x.replace('\n', ' ') for x in sentenceText]
sentenceText = [x.lower() for x in sentenceText]

#Makes a copy of the text and tokenizes it into words, removes stopwords and has all lowecase letters
wordText = nltk.word_tokenize(inputText)
stopWords = nltk.corpus.stopwords.words("english")
wordText = [c for c in wordText if c.lower() not in stopWords]
wordText = [c for c in wordText if c.isalpha()]
 
#This method is for finding the sentiment score of the text when it's split into words/sentences.
#It defines and prints the sentiment score as summation of the VADER sentiment score and the summation of positive and negative words/sentences. 
def sentimentAnalysis(text):
    sia = SentimentIntensityAnalyzer()
    val = 0.00
    pos = 0 
    neg = 0
    for x in range(len(text)):
        val += sia.polarity_scores(text[x])['compound']
        if(sia.polarity_scores(text[x])['compound'] > 0):
            pos += 1
        elif (sia.polarity_scores(text[x])['compound'] < 0) :
            neg -= 1
    print("\tSentiment Score defined as sum of positive & negative words/sentences:", pos + neg)         
    print("\tSentiment Score defined as the sum of the words/sentences given by VADER:", val)

#Uses the method to print the sentiment scores with word/sentence tokenization.
print("\nSentiment Analysis for sentences:")
sentimentAnalysis(sentenceText)
print("\nSentiment Analysis for words, with the addition of stop words:")
sentimentAnalysis(wordText)


