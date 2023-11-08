# -*- coding: utf-8 -*-
# Read text files
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


with open('.spyder-py3/Final_Project/para1.txt', 'r', encoding='utf-8') as file:
    content1 = file.read()

with open('.spyder-py3/Final_Project/para2.txt', 'r', encoding='utf-8') as file:
    content2 = file.read()


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

#Tf-Idf
def tfidf_vectorization(corpus, max_features=100):
    # Create a TfidfVectorizer object with optional max_features
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Transform the corpus into a TF-IDF representation as a dense array
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()
    
    return tfidf_matrix

# Preprocess the paragraphs
corpus1 = preprocess_paragraph(content1)
corpus2 = preprocess_paragraph(content2)

# Word2Vec model
# def train_word2vec_model(corpus, vector_size=100, window=5, min_count=1, sg=0):
#     # Tokenize the corpus to form a list of sentences
#     tokenized_corpus = [sentence.split() for sentence in corpus]
    
#     # Train the Word2Vec model
#     model = Word2Vec(tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    
#     return model

# # Train Word2Vec models for both corpus1 and corpus2
# word2vec_model1 = train_word2vec_model(corpus1, vector_size=100)
# word2vec_model2 = train_word2vec_model(corpus2, vector_size=100)

# print(word2vec_model1.wv['ancient'])
# print(word2vec_model1.wv.most_similar("ancient"))












# Bag of words usage:
X1 = bag_of_words(corpus1, max_features=100)
X2 = bag_of_words(corpus2, max_features=100)

# tf-idf usage
tfidf_matrix1 = tfidf_vectorization(corpus1, max_features=100)
tfidf_matrix2 = tfidf_vectorization(corpus2, max_features=100)

# Example usage:
# word_vectors1 = word2vec_model1.wv
# word_vectors2 = word2vec_model2.wv

# Calculate Cosine Similarity between BoW representations
bow_similarity = cosine_similarity(X1, X2)

# Calculate Cosine Similarity between TF-IDF representations
tfidf_similarity = cosine_similarity(tfidf_matrix1, tfidf_matrix2)


# Print the similarities
print("Cosine Similarity (BoW):", bow_similarity)
print("Cosine Similarity (TF-IDF):", tfidf_similarity)


