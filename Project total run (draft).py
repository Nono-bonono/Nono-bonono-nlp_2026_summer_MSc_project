#NLP assessment template 2026 - Nik Hannigan

# ----------------> uncomment line below if needed
# pip install openai

# Run __main__ at the bottom of the file, along with running parts 2 and 3.
# main(), run_part2() and run_part3() can be commented out, depending on which parts you want to run.

#############################################
# Library installation
#############################################

import nltk                               # Libraries for part 1
import pandas as pd
import spacy
import os
from pathlib import Path
import math
from collections import Counter, defaultdict
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download("cmudict")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import cmudict
cmu = cmudict.dict()

import sklearn                                            # Additional libraries for part 2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

from openai import OpenAI                             # Additional libraries for part 3
from dotenv import load_dotenv
import os
import time
from openai import RateLimitError

#############################################
# PART 1
#############################################

nlp = spacy.load("en_core_web_sm")        # Raising the input length for spaCy
nlp.max_length = 2000000

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

    for _, row in books.iterrows():                    # Iterates down through the column
        tokens = nltk.word_tokenize(row["text"])
        words = [token.lower() for token in tokens if token.isalpha()]  # Project asks for no punctuation. 
                                                                        # This is the simplest way, even if a few contractions are lost
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

def syntactic_subs(text, n = 10):
    '''Function that counts the 10 most common syntactic subjects for a text'''

    subjects = [
        token.text.lower()
        for token in text
        if token.dep_ in ('nsubj', 'nsubjpass')
    ]
    return Counter(subjects).most_common(n)

#############################################

def pmi_calc(text, subject, n = 10):
    '''
    This function takes a subject and calculates  most associated verbs by pointwise mutual information
    '''
    
    subj_verb_cnts = defaultdict(Counter)   # Setting counters for subjects, verbs and pairings
    verb_cnts = Counter()
    pairs = 0

    for token in text:
        if token.dep_ in ("nsubj", "nsubjpass"):       # Finding pairings and counting them
            subj = token.text.lower()
            verb = token.head.lemma_.lower()
            subj_verb_cnts[subj][verb] += 1
            verb_cnts[verb] += 1
            pairs += 1

    targ_cnts = subj_verb_cnts[subject]
    targ_tot = sum(targ_cnts.values())

    pmi_scores = {}
    for verb, cnt in targ_cnts.items():              # Calculating PMI
        p_subj_verb = cnt / pairs
        p_subj = targ_tot / pairs
        p_verb = verb_cnts[verb] / pairs
        pmi_scores[verb] = math.log2(p_subj_verb / (p_subj * p_verb))       # using log2 as standard

    return [(verb, round(score, 2)) for verb, score in 
        sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:n]]     # return top 10 verb pairings

#############################################

def main():
    path = Path.cwd() / "texts" 
    books = read_novels(path)
    print(books.head())

    print("TTRs:\n", nltk_ttr(books))
    print("FK scores:\n", flesch_kincaid(books))

    books_sp = parse(books)  

    for t in books_sp.itertuples():
        print(t.title, '\n', syntactic_subs(t.text), '\n')
    for t in books_sp.itertuples():
        print(t.title, '\n', pmi_calc(t.text, 'he'), '\n')
    for t in books_sp.itertuples():
        print(t.title, '\n', pmi_calc(t.text, 'she'), '\n')

#############################################
# PART 2
#############################################

def run_part2():
    '''
    Part 2 has been out behind this function to control running of the script better
    '''

    # loading and wrangling the data
    
    h10k = pd.read_csv('hansard10000.csv')
    
    h10k['party'] = h10k['party'].replace('Labour (Co-op)', 'Labour')

    h10k = h10k[h10k['party'].isin(['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat'])]
    h10k = h10k[h10k['speech_class'].isin(['Speech'])]

    h10k = h10k[h10k['speech'].str.len() > 1000]

    print('Shape of the dataframe:', h10k.shape)


# Initial vectorizing and setup
#############################################

    # Vectorizing the data and setting up Random Forest and SVM

    vec_h10k = TfidfVectorizer(stop_words = 'english', max_features = 3000)            # Setting parameters for vectors
    x = vec_h10k.fit_transform(h10k['speech'])
    y = h10k['party']

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, random_state = 26)       # Splitting the set

    rando = RandomForestClassifier(n_estimators = 300, random_state = 26).fit(x_train, y_train)      # Random Forest
    support_vm = SVC(kernel = 'linear', random_state = 26).fit(x_train, y_train)                     # Support Vector Machine


    # Classification of parties and F1 score

    for name, clf in [('RandomForest', rando), ('SVM', support_vm)]:
        party_pred = clf.predict(x_test)
        print(name)
        print('Macro avg. F1:', f1_score(y_test, party_pred, average = 'macro', zero_division = 0))
        print(classification_report(y_test, party_pred, zero_division = 0))


# Vectorizing and setup with n-grams
#############################################

    # Vectorizing the data and setting up Random Forest and SVM with uni-, bi- and trigrams as parameters.

    vec_h10k_ngram = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range = (1, 3))     # Setting parameters for vectors
    x1 = vec_h10k_ngram.fit_transform(h10k['speech'])
    y1 = h10k['party']

    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, stratify = y1, random_state = 26)       # Splitting the set

    rando1 = RandomForestClassifier(n_estimators = 300, random_state = 26).fit(x1_train, y1_train)      # Random Forest
    support_vm1 = SVC(kernel = 'linear', random_state = 26).fit(x1_train, y1_train)                     # Support Vector Machine

    # Classification of parties and F1 score

    for name1, clf in [('RandomForest', rando1), ('SVM', support_vm1)]:
        party_pred1 = clf.predict(x1_test)
        print(name1)
        print('Macro avg. F1:', f1_score(y1_test, party_pred1, average = 'macro', zero_division = 0))
        print(classification_report(y1_test, party_pred1, zero_division = 0))


# Vectorizing and setup with custom tokenizer
#############################################

    lemmatizer = WordNetLemmatizer()                  # Setting some variables for the function
    stop_w = set(stopwords.words('english'))

    def custom_toke(text):
        ''' 
        Takes words and tokenizes them 
        First puts all letters into lowercase, then drops anything that's not letters
        Lemmatizes for consistency across similar words
        Removes stopwords
        '''

        text = text.lower()
        toke = re.findall(r'[a-z]+', text)
        toke = [t for t in toke if len(t) > 2]
        toke = [lemmatizer.lemmatize(t, pos = 'v') for t in toke]     # Adding verbs into lemmatizer mix
        toke = [t for t in toke if t not in stop_w]
    
        return toke

    # Testing out the new tokenizer

    vec_h10k_custom = TfidfVectorizer(tokenizer = custom_toke, token_pattern = None, max_features = 2000)       # Setting parameters for vectors

    x2 = vec_h10k_custom.fit_transform(h10k['speech'])
    y2 = h10k['party']

    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, stratify = y2, random_state = 26)       # Splitting the set

    rando2 = RandomForestClassifier(n_estimators = 300, random_state = 26).fit(x2_train, y2_train)      # Random Forest
    support_vm2 = SVC(kernel = 'linear', random_state = 26).fit(x2_train, y2_train)                     # Support Vector Machine

    # Classification of parties and F1 score

    for name2, clf in [('RandomForest', rando2), ('SVM', support_vm2)]:
        party_pred2 = clf.predict(x2_test)
        print(name2)
        print('Macro avg. F1:', f1_score(y2_test, party_pred2, average = 'macro', zero_division = 0))
        print(classification_report(y2_test, party_pred2, zero_division = 0))


#############################################
# PART 3
#############################################

def run_part3():'
    '''
    Part 2 has been out behind this function to control running of the script better
    '''
    # Loading API key through a .env file. The .env file is not uploaded to github.

    load_dotenv('the_nlp.env')  
    api_key = os.getenv("OPENROUTER_API_KEY")                 # With credit

    if not api_key:                                           # Raises error in case of missing API key
        raise RuntimeError("API_KEY not found, check the_nlp.env")
        
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Importing and checking speeches
    # Wrangling data in line with previous imports
    ########################################

    h5 = pd.read_csv('hansard500.csv')

    h5['party'] = h5['party'].replace('Labour (Co-op)', 'Labour')
    h5 = h5[h5['speech_class'].isin(['Speech'])]
    h5 = h5[h5['party'].isin(['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat'])]

    h5.shape

    x = h5['speech']                 # defining x and y
    y = h5['party']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, random_state=26)                    # Same seed as part 2 for consistency

    labels = sorted(y.unique().tolist())                      # Creating a list of party labels to feed into LLM
    label_str = ", ".join(labels)

# function to talk to LLM
# Built in rate limit error catcher

    def classify(prompt, max_retries=5):
        '''
        This function takes a prompt from the user and feeds it into the connected LLM
        Initially I was using a free version and hitting rate limits a lot so it is designed to notify if there are rate
        limit issues.
        I decided to keep the rate limit checker in as it is still useful to see if there is an issue.
        '''
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,                                      # Temperature set at 0 as I want the LLM to be as unimaginative as possible
                    max_tokens=10,                                      # Limiting tokens to limit answer, we only want a single label
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                wait = 30 * (attempt + 1)   
                print(f"Rate-limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
        raise RuntimeError("Still rate-limited after retries")

# Function to normalize output

    def normalize(out):
        '''This is a function to normalize output from the LLM and maintain consistency'''
        
        out = out.lower()
        for lab in labels:
            if lab.lower() in out:
                return lab

    def build_examples(n_per_party=1):
        '''Function to build some examples for the few shot process. All taken from the training data'''
        
        shots = []
        for lab in labels:
            idx = y_train[y_train == lab].index[:n_per_party]
            for i in idx:
                shots.append((x_train.loc[i][:600], lab))
        return shots

    example_block = "\n\n".join(
        f"Speech: {sp}\nParty: {lab}" for sp, lab in build_examples())

    ZERO_SHOT_PROMPT = """Classify UK parliamentary speeches by the speaker's political party.
    Read the speech and respond with exactly one label from this list: ['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat']
    Respond with the label only. Do not add an explanation or punctuation. There should be nothing beyond the label.

    Speech:
    {speech}

    Party:""" 


    FEW_SHOT_PROMPT = """Classify UK parliamentary speeches by the speaker's political party.
    Read the speech and respond with exactly one label from this list: ['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat']
    Respond with the label only. Do not add an explanation or punctuation. There should be nothing beyond the label.

    Here are some labelled examples:

    {examples}

    Now classify this speech. Respond with the label only.

    Speech:
    {speech}

    Party:"""


    def run(prompt_template, few_shot=False):
        '''
        This function wraps around the 'classify()' function and reurtns predictions for the political speeches
        '''
    
        preds = []
        for i, s in enumerate(x_test):
            if few_shot:
                p = prompt_template.format(examples=example_block, speech=s[:3000])
            else:
                p = prompt_template.format(speech=s[:4000])
            preds.append(normalize(classify(p)))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(x_test)} done")            # tracks progress
            time.sleep(3.5)                                     # stay under 20 req/min to regulate usage
        return preds

    def score(name, preds):
        '''
        Prints out formatted scores from LLM answer
        '''
    
        print(f"\n===== {name} =====")
        print("Macro-avg F1:", f1_score(y_test, preds, average='macro', zero_division=0))
        print(classification_report(y_test, preds, zero_division=0))

    # Zero shot prompt results
    zero_preds = run(ZERO_SHOT_PROMPT, few_shot=False)
    score("ZERO-SHOT", zero_preds)

    # Few shot prompt results
    few_preds  = run(FEW_SHOT_PROMPT, few_shot=True)
    score("FEW-SHOT", few_preds)

#############################################

if __name__ == "__main__":
    main()
    run_part2()
    run_part3()