from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Calculate TF-IDF
def calculate_tfidf(documents):
    vectorizer = TfidfVectorizer(tokenizer=preprocess)
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix.toarray()

# Calculate Word Embeddings
def calculate_word_embeddings(documents):
    tokenized_documents = [preprocess(doc) for doc in documents]
    model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, sg=0)
    word_vectors = model.wv
    return word_vectors

# Calculate Cosine Similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm_vector1 = sum(a * 2 for a in vector1) * 0.5
    norm_vector2 = sum(a * 2 for a in vector2) * 0.5
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# Read text files
file_path1 = '.spyder-py3/Final_Project/para1.txt'

file_path2 =  '.spyder-py3/Final_Project/para2.txt'

with open(file_path1, 'r') as file:
    document1 = file.read()

with open(file_path2, 'r') as file:
    document2 = file.read()

documents = [document1, document2]

# Calculate TF-IDF matrix
tfidf_matrix = calculate_tfidf(documents)

# Calculate Word Embeddings
word_embeddings = calculate_word_embeddings(documents)

# Calculate similarity using TF-IDF
tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

# Calculate similarity using Word Embeddings
embeddings_similarity = word_embeddings.similarity('dog','cat')  # Adjust words as needed

print("TF-IDF Similarity:", tfidf_similarity)
print("Word Embedding Similarity:", embeddings_similarity)