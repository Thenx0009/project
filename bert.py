# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:29:50 2023

@author: INDIA
"""
import fitz
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import fasttext
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

# Download the stopwords resource
nltk.download('stopwords')

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# Specify the paths to your PDF files
pdf_file_path1 = '/content/act.pdf'
pdf_file_path2 = '/content/policy.pdf'

# Read text from PDF files
content1 = read_pdf(pdf_file_path1)
content2 = read_pdf(pdf_file_path2)

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


# Preprocess the paragraphs
corpus1 = preprocess_paragraph(content1)
corpus2 = preprocess_paragraph(content2)

def save_sentences_to_file(sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

# Save preprocessed sentences to temporary files
temp_file_path1 = '/content/sample_data/temp_corpus1.txt'
temp_file_path2 = '/content/sample_data/temp_corpus2.txt'

save_sentences_to_file(corpus1, temp_file_path1)
save_sentences_to_file(corpus2, temp_file_path2)



# Function to train a custom FastText model
def train_custom_fasttext_model(file_path, model_save_path, dim=100, min_count=1, epoch=5):
    try:
        model = fasttext.train_unsupervised(
            file_path,
            dim=dim,
            minCount=min_count,
            epoch=epoch
        )

        # Save the trained model
        model.save_model(model_save_path)
        print(f"Model saved successfully at: {model_save_path}")

        return model
    except Exception as e:
        print("Error during model training:", str(e))
        return None

# Train FastText model on your custom data
custom_fasttext_model_path = '/content/sample_data/custom_model.bin'

# Train FastText model using file paths
custom_fasttext_model = train_custom_fasttext_model(temp_file_path1, custom_fasttext_model_path)

# Check if the model is not None before getting vectors
if custom_fasttext_model:
    # Get vectors for each sentence in the corpus using the trained model
    custom_fasttext_vectors1 = [custom_fasttext_model.get_sentence_vector(sentence) for sentence in corpus1]
    custom_fasttext_vectors2 = [custom_fasttext_model.get_sentence_vector(sentence) for sentence in corpus2]

    # Calculate Cosine Similarity between custom FastText representations
    custom_fasttext_similarity = cosine_similarity(custom_fasttext_vectors1, custom_fasttext_vectors2)

    # Print the similarities
    print("Cosine Similarity (Custom FastText):", custom_fasttext_similarity)
else:
    print("Custom FastText model is None. Please check the training process.")