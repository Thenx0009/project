# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:44:59 2023

@author: INDIA
"""

import fitz
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import fasttext
from gensim.models import KeyedVectors
from glove import Corpus, Glove




def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# # Specify the paths to your PDF files
pdf_file_path1 = '.spyder-py3/Final_Project/act.pdf'
pdf_file_path2 = '.spyder-py3/Final_Project/policy.pdf'

# Read text from PDF files
content1 = read_pdf(pdf_file_path1)
content2 = read_pdf(pdf_file_path2)

# Function to preprocess a paragraph
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

        # Join the cleaned words to form a sentence
        cleaned_sentence = ' '.join(filtered_words)

        corpus.append(cleaned_sentence)

    return corpus

# Preprocess the paragraphs
corpus1 = preprocess_paragraph(content1)
corpus2 = preprocess_paragraph(content2)


# Function to train GloVe model
def train_glove_model(corpus, vector_size=50, window_size=10):
    glove_corpus = Corpus()
    
    # Fit the corpus to the GloVe model
    glove_corpus.fit(corpus, window=window_size)
    
    # Create a Glove model
    glove_model = Glove(no_components=vector_size, learning_rate=0.05)
    
    # Fit the GloVe model on the corpus
    glove_model.fit(glove_corpus.matrix, epochs=30, no_threads=4, verbose=True)
    
    # Add the learned words to the model's vocabulary
    glove_model.add_dictionary(glove_corpus.dictionary)
    
    return glove_model

# Train GloVe models on your corpora
glove_model1 = train_glove_model(corpus1)
glove_model2 = train_glove_model(corpus2)

# Get the word vectors from the trained GloVe models
word_vectors1 = glove_model1.word_vectors
word_vectors2 = glove_model2.word_vectors

# Calculate Cosine Similarity between GloVe word vectors
glove_similarity = cosine_similarity(word_vectors1, word_vectors2)

# Print the similarity
print("Cosine Similarity (GloVe):", glove_similarity)
