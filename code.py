# -*- coding: utf-8 -*-
# Read text files
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

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# Specify the paths to your PDF files
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

# Bag of words
def bag_of_words(corpus, max_features=100):
    # Create a CountVectorizer object with the specified max_features
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()

    return X

# Tf-Idf
def tfidf_vectorization(corpus, max_features=100):
    # Create a TfidfVectorizer object with optional max_features
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Transform the corpus into a TF-IDF representation as a dense array
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()

    return tfidf_matrix

# FastText
def get_fasttext_vectors(corpus, model_path, min_count=1):
    # Train a FastText model
    model = fasttext.train_unsupervised(model_path, dim=100, minCount=min_count)

    # Get vectors for each sentence in the corpus
    vectors = [model.get_sentence_vector(sentence) for sentence in corpus]

    return vectors


# Preprocess the paragraphs
corpus1 = preprocess_paragraph(content1)
corpus2 = preprocess_paragraph(content2)

# Bag of words usage:
# X1 = bag_of_words(corpus1, max_features=100)
# X2 = bag_of_words(corpus2, max_features=100)

# Tf-idf usage
# tfidf_matrix1 = tfidf_vectorization(corpus1, max_features=100)
# tfidf_matrix2 = tfidf_vectorization(corpus2, max_features=100)

# FastText usage
# Provide the path to your FastText model
fasttext_model_path = 'C:\\Users\\INDIA\\fasttext_wheel\\cc.en.300.bin'
fasttext_vectors1 = get_fasttext_vectors(corpus1, fasttext_model_path, min_count=1)
fasttext_vectors2 = get_fasttext_vectors(corpus2, fasttext_model_path, min_count=1)


# Example usage:
# word_vectors1 = word2vec_model1.wv
# word_vectors2 = word2vec_model2.wv

# Calculate Cosine Similarity between BoW representations
# bow_similarity = cosine_similarity(X1, X2)

# # Calculate Cosine Similarity between TF-IDF representations
# tfidf_similarity = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Calculate Cosine Similarity between FastText representations
fasttext_similarity = cosine_similarity(fasttext_vectors1, fasttext_vectors2)

# Print the similarities
# print("Cosine Similarity (BoW):", bow_similarity)
# print("Cosine Similarity (TF-IDF):", tfidf_similarity)
print("Cosine Similarity (FastText):", fasttext_similarity)

