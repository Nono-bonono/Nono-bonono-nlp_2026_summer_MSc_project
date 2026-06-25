#NLP assessment template 2026

import nltk                               # Import necessary libraries
from nltk.corpus import cmudict
cmu = cmudict.dict()
import pandas as pd
import spacy
import os
from pathlib import Path
import math
from collections import Counter

nlp = spacy.load("en_core_web_sm")        # Raising the input length for spaCy
nlp.max_length = 2000000

#############################################

def read_novels(text_dir="texts"):
    '''
    Function that takes the .txt book files from local 'texts' directory and reads them into a Dataframe

    Order the resulting dataframe by publishing year, returns the DF.
    '''
    rows = []

    for filename in os.listdir(text_dir):
        if not filename.endswith(".txt"):                           # Ignores files that aren't .txt
            continue

        title, author, year = filename[:-len(".txt")].split("-")    # splits the filename off from the file type

        with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()

        rows.append({
            "text": text,
            "title": title.replace("_", " "),                       # Some formatting for the titles
            "author": author,
            "year": int(year),
        })

    books = pd.DataFrame(rows, columns=["text", "title", "author", "year"])          # Create dataframe and titles
    books = books.sort_values(by="year", ignore_index=True)                          # Sort by year and reset index

    return books

#############################################

def nltk_ttr(books):
    '''Type token ratio function, returns a dictionary'''
    
    ttrs = {}

    for _, row in books.iterrows():
        tokens = nltk.word_tokenize(row["text"])
        words = [token.lower() for token in tokens if token.isalpha()]  # Project asks for no punctuation. This is the simplest way, even if a few contractions are lost
        ttrs[row["title"]] = len(set(words)) / len(words)               # Actual ratio calculation

    return ttrs

#############################################

def count_syllables(word):
    '''Function to count syllables'''
    word = word.lower()

    if word in cmu:
        return len([phoneme for phoneme in cmu[word][0] if phoneme[-1].isdigit()])

    return len(re.findall(r"[aeiouy]+", word))

#############################################

def flesch_kincaid(books):
    '''Calculating the Flesch Kinkaid score'''
    
    fk_scores = {}

    for _, row in books.iterrows():
        text = row["text"]

        sentences = nltk.sent_tokenize(text)                                                   # Tokenizing sentences
        words = [token.lower() for token in nltk.word_tokenize(text) if token.isalpha()]       # Tokenizing words

        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(count_syllables(word) for word in words)

        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)    # FK reading level score formula
        fk_scores[row["title"]] = score

    return fk_scores

#############################################

def parse(df, store_path="texts/parsed.pickle"):
    '''
    This function takes a dataframe and pickles it after processing it
    '''
    
    df["parsed"] = df["text"].apply(nlp)
    df.to_pickle(store_path)

    return df

#############################################

#############################################



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # path = Path.cwd() / "texts" / "novels"
    # print(path)
    # df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # call functions for part (e) here.
