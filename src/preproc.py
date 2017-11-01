# read in some helpful libraries
import nltk # the natural langauage toolkit, open-source NLP
import pandas as pd # dataframes

### Read in the data

# read our data into a dataframe
texts = pd.read_csv("train.csv")

# look at the first few rows
print(texts.head())


### Split data

# split the data by author
byAuthor = texts.groupby("author")

### Tokenize (split into individual words) our text

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

# for each author...
for name, group in byAuthor:
    # get all of the sentences they wrote and collapse them into a
    # single long string
    sentences = group['text'].str.cat(sep = ' ')
    
    print "Sentences for ", name, " are ", sentences.decode("utf-8")[:100]

    # convert everything to lower case (so "The" and "the" get counted as 
    # the same word rather than two different words)
    sentences = sentences.lower()
    
    # split the text into individual tokens    
    tokens = nltk.tokenize.word_tokenize(sentences.decode("utf-8"))

    # calculate the frequency of each token
    frequency = nltk.FreqDist(tokens)

    # add the frequencies for each author to our dictionary
    wordFreqByAuthor[name] = (frequency)

# now we have an dictionary where each entry is the frequency distribution
# of words for a specific author.     

# see how often each author says "blood"
for author in wordFreqByAuthor.keys():
    print("Author is: " + author)
    for word in wordFreqByAuthor[author]:
        print("Word ", word, " is ", wordFreqByAuthor[author].freq(word))


