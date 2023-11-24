# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:39:51 2023

@author: INDIA
"""

import os
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize global variables
corpus1 = []
corpus2 = []

#Fuction to preprocess the paragraph
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

        # Only append non-empty sentences to the corpus
        if cleaned_sentence:
            corpus.append(cleaned_sentence)

    return corpus
# Common function to load and preprocess files in a folder
def load_and_preprocess_folder(folder_path):
    corpus = []
    valid_extensions = ('.pdf')
    selected_files = [file for file in os.listdir(folder_path) if file.lower().endswith(valid_extensions)]

    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        content = ""

        # Process PDF files
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"

        # Preprocess the content and add it to the corpus list
        corpus.extend(preprocess_paragraph(content))

    return corpus

# Example usage of the functions
# (Note: You need to manually set the folder paths in Colab)

# Load and preprocess data from the first folder
folder_path1 = "/path/to/your/first/folder"
corpus1 = load_and_preprocess_folder(folder_path1)

# Load and preprocess data from the second folder
folder_path2 = "/path/to/your/second/folder"
corpus2 = load_and_preprocess_folder(folder_path2)

# Print the content of the corpora
print("Corpus 1:")
print(corpus1)

print("\nCorpus 2:")
print(corpus2)