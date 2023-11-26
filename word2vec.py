# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:42:04 2023

@author: INDIA
"""

import nltk
import os
import numpy as np
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Initialize global variables
corpus1 = []
corpus2 = []

# Function to preprocess the paragraph
def preprocess_paragraph(paragraph):
    lemmatizer = WordNetLemmatizer()

    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    corpus = []
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Remove non-alphabetic characters, convert to lowercase, and lemmatize
        clean_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]

        # Remove stopwords
        filtered_words = [word for word in clean_words if word not in stopwords.words('english')]

        # Only append non-empty lists of words to the corpus
        if filtered_words:
            corpus.append(filtered_words)

    return corpus

# Common function to load and preprocess files in a folder
def load_and_preprocess_folder(folder_path):
    corpus = []
    valid_extensions = ('.pdf')

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file_name)
                content = ""

                # Process PDF files
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"

                # Preprocess the content and add it to the corpus list
                corpus.extend(preprocess_paragraph(content))

    return corpus

# Load and preprocess data from the first folder
folder_path1 = "C:/Users/INDIA/.spyder-py3/FINAL_PROJECT/sample_acts"
corpus1 = load_and_preprocess_folder(folder_path1)

# Load and preprocess data from the second folder
folder_path2 = "C:/Users/INDIA/.spyder-py3/FINAL_PROJECT/sample_policy"
corpus2 = load_and_preprocess_folder(folder_path2)


# # Print the content of the corpora
# print("Corpus 1:")
# print(corpus1)

# print("\nCorpus 2:")
# print(corpus2)


# Function to train Word2Vec model
def train_word2vec_model(corpus):
    return Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Function to get vectors from a Word2Vec model
def get_vectors(model, sentences, vector_size=100):
    vectors = [
        [model.wv.get_vector(word) for word in sentence if word in model.wv]
        for sentence in sentences
    ]

    # Ensure vectors have the same maximum length
    max_length = max(map(len, vectors))
    vectors_padded = np.array([
        sentence + [np.zeros(vector_size)] * (max_length - len(sentence))
        for sentence in vectors
    ])

    return vectors_padded

# Train Word2Vec model
word2vec_model1 = train_word2vec_model(corpus1)
word2vec_model2 = train_word2vec_model(corpus2)

# Get vectors for each sentence
vectors1 = get_vectors(word2vec_model1, corpus1)
vectors2 = get_vectors(word2vec_model2, corpus2)

# Ensure vectors have the same maximum length across vectors1 and vectors2
max_length1 = max(len(sentence) for sentence in vectors1)
max_length2 = max(len(sentence) for sentence in vectors2)
max_length = max(max_length1, max_length2)
# Ensure all sentence vectors have the same maximum length
max_sentence_length = max(len(sentence) for sentence in corpus1 + corpus2)
def get_padded_vectors(model, sentences, vector_size=100):
    vectors = [
        [model.wv.get_vector(word) for word in sentence if word in model.wv]
        for sentence in sentences
    ]

    # Pad all vectors to the same length
    padded_vectors = np.array([
        sentence + [np.zeros(vector_size)] * (max_sentence_length - len(sentence))
        for sentence in vectors
    ])

    return padded_vectors
# Get padded vectors for each corpus
vectors1_padded = get_padded_vectors(word2vec_model1, corpus1)
vectors2_padded = get_padded_vectors(word2vec_model2, corpus2)

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(vectors1_padded, vectors2_padded)

# Print the cosine similarity matrix for the first 20 rows
print("Cosine Similarity (Word2Vec):")
print(cosine_sim_matrix[:20, :20])