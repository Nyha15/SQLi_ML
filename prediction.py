import pickle
import re
import pandas as pd
import numpy as np
from collections import Counter
import math

import os

#CWD to get the files
CWD_ = os.getcwd()

#Get all the file paths you need for deployment
RFC_FE_model_path = os.path.join(CWD_,'model','FE_RFC_model_final.model')
TFIDF_vectorizer_path = os.path.join(CWD_,'model','tfidf_vect_final.sav')
reservedWords_path = os.path.join(CWD_,'data','reswrds_final.csv')

# Load the files from the paths
with open(RFC_FE_model_path, 'rb') as file:
	RFC_FE_model_final = pickle.load(file)
with open(TFIDF_vectorizer_path, 'rb') as file:
	TFIDF_vectorizer_final = pickle.load(file)
reservedWords = pd.read_csv("reservedWords_path", dtype=str)

def clean_query(query):

    """
    This function cleans the input query by removing all the extra whitespaces
    and then converts the entire query to upper case and strips any leading/trailing
    spaces.
  	"""
    cleaned_query = re.sub(' +', ' ', query)
    return cleaned_query.upper().strip()

reserved_words_list = reservedWords['ReservedWord'].tolist()

def isReservedWord(word):
        """
        This function checks if the given word is a reseved SQL word or not.
      	"""
        return word in reserved_words_list

def has_reserved_words(query):

    words = query.split()
    return any(list(map(isReservedWord, words)))

def complete(query):
    """
      This function checks if the query is a complete query or not.
    """
    for word in reservedWords['ReservedWord'].apply(str):
        if query.startswith(word):
          if not (query.endswith("'") or query.endswith('"')):
              return True
    return False

def get_entropy(query):
    """
            This function calculates the entropy.
            Entropy is the probability of occurence of certain words in the query
    """
    prob, l = Counter(query.split()), float(len(query.split()))
    return -sum( count/l * math.log(count/l, 2) for count in prob.values())

def predict(query):
    """
        This function predicts the class label and returns the class label or
        probability based on the request.
    """
    cleaned_query = clean_query(query)
    no_of_special_chars = len(re.findall('[^a-zA-Z0-9\s]',query))
    query_length = len(cleaned_query.split())
    pattern = len(re.findall(r'\d\s*=\s*\d',cleaned_query))
    contains_keywords = has_reserved_words(cleaned_query)
    complete_query = complete(cleaned_query)
    single_word = cleaned_query.split() == 1
    entropy = get_entropy(cleaned_query)
    transformed_query = TFIDF_vectorizer_final.transform([cleaned_query])
    tfidf_vect = pd.DataFrame(data=transformed_query.toarray(),columns=TFIDF_vectorizer_final.get_feature_names())
    tfidf_vect['sc_length'] = no_of_special_chars
    tfidf_vect['query_length'] = query_length
    tfidf_vect['pattern'] = pattern
    tfidf_vect['contains_keywords'] = contains_keywords
    tfidf_vect['complete_query'] = complete_query
    tfidf_vect['single_word'] = single_word
    tfidf_vect['entropy'] = entropy
    probability = RFC_FE_model_final.predict_proba(tfidf_vect)
    predicted_class = RFC_FE_model_final.predict(tfidf_vect)
    return probability[0]

def predict_probability(input_query):
  if isinstance(input_query, str):
    return predict(input_query)
