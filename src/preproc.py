# read in some helpful libraries
import nltk # the natural langauage toolkit, open-source NLP
import pandas as pd # dataframes

from nltk.corpus import stopwords # list of stopwords
from nltk.stem import WordNetLemmatizer # lemmatizer


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

### Tokenize (split into individual words) our text

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

# for each author...
for name, group in byAuthor:
    # get all of the sentences they wrote and collapse them into a
    # single long string
    sentences = group['text'].str.cat(sep = ' ')
    
    print("Sentences for ", name, " are ", sentences[:100])

    # convert everything to lower case (so "The" and "the" get counted as 
    # the same word rather than two different words)
    sentences = sentences.lower()
    
    # split the text into individual tokens    
    tokens = nltk.tokenize.word_tokenize(sentences)

    print(type(tokens))

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

# now we have an dictionary where each entry is the frequency distribution
# of words for a specific author.     

# save the frequencies in a list for every author and sort them
wordFreqDistrByAuthor = {}

for author in wordFreqByAuthor.keys():

    wordFreqDistrByAuthor[author] = []

    for word in wordFreqByAuthor[author]:
        wordFreqDistrByAuthor[author].append([word, wordFreqByAuthor[author].freq(word)])

    wordFreqDistrByAuthor[author] = sorted(wordFreqDistrByAuthor[author], key=lambda x: x[1], reverse=True)

    print(author, wordFreqDistrByAuthor[author][:15])
    print("\n\n")


