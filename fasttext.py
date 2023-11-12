# import fitz
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
# import fasttext
# from gensim.models import KeyedVectors
# from sklearn.model_selection import train_test_split

# def read_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ''
#     for page_num in range(doc.page_count):
#         page = doc[page_num]
#         text += page.get_text()
#     doc.close()
#     return text

# # Specify the paths to your PDF files
# pdf_file_path1 = '.spyder-py3/Final_Project/act.pdf'
# pdf_file_path2 = '.spyder-py3/Final_Project/policy.pdf'

# # Read text from PDF files
# content1 = read_pdf(pdf_file_path1)
# content2 = read_pdf(pdf_file_path2)

# # Function to preprocess a paragraph
# def preprocess_paragraph(paragraph):
#     lemmatizer = WordNetLemmatizer()

#     # Tokenize the paragraph into sentences
#     sentences = sent_tokenize(paragraph)

#     corpus = []
#     for sentence in sentences:
#         # Tokenize the sentence into words
#         words = word_tokenize(sentence)

#         # Remove non-alphabetic characters, convert to lowercase, and lemmatize
#         clean_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]

#         # Remove stopwords
#         filtered_words = [word for word in clean_words if word not in stopwords.words('english')]

#         # Join the cleaned words to form a sentence
#         cleaned_sentence = ' '.join(filtered_words)

#         corpus.append(cleaned_sentence)

#     return corpus

# # Preprocess the paragraphs
# corpus1 = preprocess_paragraph(content1)
# corpus2 = preprocess_paragraph(content2)




# # Function to train a custom FastText model
# def train_custom_fasttext_model(corpus, model_save_path, dim=100, min_count=1, epoch=5):
#     try:
#         # Check if the corpus is not empty
#         if corpus:
#             model = fasttext.train_unsupervised(
#                 corpus,
#                 dim=dim,
#                 minCount=min_count,
#                 epoch=epoch
#             )

#             # Save the trained model
#             model.save_model(model_save_path)
#             print(f"Model saved successfully at: {model_save_path}")

#             return model
#         else:
#             print("Error: Empty corpus provided for model training.")
#             return None
#     except Exception as e:
#         print("Error during model training:", str(e))
#         return None

# # Train FastText model on your custom data
# # Specify the complete path, including the model file name
# custom_fasttext_model_path = 'C:\\Users\\INDIA\\fasttext_models\\custom_model.bin'

# # Train FastText model on your custom data
# custom_fasttext_model = train_custom_fasttext_model(corpus1 + corpus2, custom_fasttext_model_path)

# # Check if the model is not None before getting vectors
# if custom_fasttext_model:
#     # Get vectors for each sentence in the corpus using the trained model
#     custom_fasttext_vectors1 = [custom_fasttext_model.get_sentence_vector(sentence) for sentence in corpus1]
#     custom_fasttext_vectors2 = [custom_fasttext_model.get_sentence_vector(sentence) for sentence in corpus2]

#     # Calculate Cosine Similarity between custom FastText representations
#     custom_fasttext_similarity = cosine_similarity(custom_fasttext_vectors1, custom_fasttext_vectors2)

#     # Print the similarities
#     print("Cosine Similarity (Custom FastText):", custom_fasttext_similarity)
# else:
#     print("Custom FastText model is None. Please check the training process.")

import fasttext
print(fasttext.__version__)
