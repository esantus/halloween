# read in some helpful libraries
import numpy as np
import nltk # the natural langauage toolkit, open-source NLP
import pandas as pd # dataframes
import glob
import os

from nltk.corpus import stopwords # list of stopwords
from nltk.stem import WordNetLemmatizer # lemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten

word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

### Import all stopwords for English and initiate the lemmatizer

stopwords = stopwords.words('english')
with open("../datasets/stopwords-en.txt", "r", encoding='UTF8') as stpw:
    for word in stpw:
        stopwords.append(word.strip())

stopwords = set(stopwords)

wordnet_lemmatizer = WordNetLemmatizer()

#print stopwords

### Read in the data

# read our data into a dataframe
texts = pd.read_csv("../datasets/train.csv")

# look at the first few rows
print(texts.head())

### Split data

# split the data by author
byAuthor = texts.groupby("author")

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()


def LexicalFeatures():
    ### Tokenize (split into individual words) our text
    #initialize vectors
    fvs_lexical = np.zeros((len(byAuthor), 3), np.float64)
    fvs_punct = np.zeros((len(byAuthor), 3), np.float64)
    # for each author...
    e = -1
    for name, group in byAuthor:
        e = e + 1
        # get all of the sentences they wrote and collapse them into a
        # single long string
        sentences = group['text'].str.cat(sep = ' ')
      
        print("Sentences for ", name, " are ", sentences[:100])

        # convert everything to lower case (so "The" and "the" get counted as 
        # the same word rather than two different words)
        sentences = sentences.lower()    
        
        # split the text into individual tokens    
        tokens = nltk.tokenize.word_tokenize(sentences)
        # Commas per sentence    
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # Semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # Colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))    
        
        words = word_tokenizer.tokenize(sentences)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])
        vocab = set(words)
        
        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))    
        
        # lemmatize the text
        new_tokens = []

        for token in tokens:
            if token not in stopwords and token not in ["n't", "though", "never", "ever", "even", "however", "us", "one", "yet", "\'", "could", "would", "``", "?", "'s", "\'\'", ".", ",", ":", "!", ";", ")", "(", ",", "[", "]", "{", "}", "#", "###", "##", "|", "*", "@card@", "..", "..."]:
                new_tokens.append(wordnet_lemmatizer.lemmatize(token))
        
        tokens = new_tokens

        # calculate the frequency of each token
        frequency = nltk.FreqDist(tokens)

        # add the frequencies for each author to our dictionary
        wordFreqByAuthor[name] = (frequency)

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)
    fvs_punct = whiten(fvs_punct)   

    #return fvs_lexical, fvs_punct, wordFreqByAuthor
    return fvs_lexical, fvs_punct


def BagOfWords():

    # save the frequencies in a list for every author and sort them
    wordFreqDistrByAuthor = {}

    for author in wordFreqByAuthor.keys():

        wordFreqDistrByAuthor[author] = []

        for word in wordFreqByAuthor[author]:
            wordFreqDistrByAuthor[author].append([word, wordFreqByAuthor[author].freq(word)])

        wordFreqDistrByAuthor[author] = sorted(wordFreqDistrByAuthor[author], key=lambda x: x[1], reverse=True)

        print(author, wordFreqDistrByAuthor[author][:15])
        print("\n\n")

    return wordFreqDistrByAuthor


def PredictAuthors(fvs):
    """
    Use k-means clustering to fit a model
    """
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, verbose=0)
    km.fit(fvs)

    return km


if __name__ == '__main__':
    
    feature_sets = list(LexicalFeatures())
    #feature_sets.append(BagOfWords())

    classifications = [PredictAuthors(fvs).labels_ for fvs in feature_sets]
    #for results in classifications:
    #    # in our case, we know the author of chapter 10, so set it to
    #    if results[2] == 0: results = 1 - results
    #    print(' '.join([str(a) for a in results]))
    print("classifications", classifications)

   
