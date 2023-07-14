# packages
import spacy
import nltk
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def clean_all(df, col_name):

    # df, col_name = df, 'skill_description'

    # encode for only ascii characters
    df[col_name] = df[col_name].map(ascii_rm)
    # df.iloc[0][0]
    # lowercase texts
    df[col_name] = df[col_name].map(lambda x: x.lower())

    # lemmatize words (transform running, ran to run)
    df[col_name] = df[col_name].astype(str).map(lemma)

    # remove punctuation: entferne kommas, slash zeichen usw.
    df[col_name] = df[col_name].map(punc_n)

    return df

def clean_skills(df, col_name):

    # df, col_name = reviews, 'reviewText'

    # lowercase texts
    df[col_name] = df[col_name].map(lambda x: x.lower())

    # remove punctuation: entferne kommas, slash zeichen usw.
    df[col_name] = df[col_name].map(punc_n)

    return df


def ascii_rm(comment):

    if type(comment) != str:
        comment = str(comment)
        comment = comment.encode('ascii', errors='ignore')
    else:
        comment = comment.encode('ascii', errors='ignore')

    return comment


def get_language(text):

    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key=lambda x: x[1])[0]


def punc_n(comment):

    regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~') + '0-9\\r\\t\\n]')
    # comment = df.iloc[0][0]
    nopunct = regex.sub(" ", comment)
    nopunct_words = nopunct.split(' ')
    filter_words = [word.strip() for word in nopunct_words if word != '']
    words = ' '.join(filter_words)
    return words


def lemma(comment):
    # comment = df.iloc[0][0].astype(str)
    # lemmatized = df[col_name].astype(str).map(nlp)
    # print(type(comment))
    lemmatized = nlp(comment)
    lemmatized_final = ' '.join([word.lemma_ if word.text != 'aws' else word.text for word in lemmatized if word.lemma_ != '\'s'])
    return lemmatized_final

#nlp = spacy.load('en_core_web_sm')
#doc = nlp("aws ec2 instances for application deployment  as a devops engineer, i need to provision aws ec2 instances to host our application, ensuring scalability, high availability, and efficient resource allocation. this task involves selecting appropriate instance types, configuring security groups and network settings, and setting up auto-scaling policies to handle varying application workloads.  '")

#lemmatized_final = ' '.join([word.lemma_ for word in doc if word.lemma_ != '\'s'])



#' '.join([word.lemma_ for word in lemmatized if word.lemma_ != '\'s'])
#for word in doc:
    # print(word.lemma_)
#    print(word.lemma_ != '\'s')
    # word = 'workloads'
    #if str(word) != 'aws' and word.lemma_ != '\'s':
#    lemmatized_final = ' '.join([word.lemma_])