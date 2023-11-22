import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):

    words = word_tokenize(text)

    # Remove punctuation
    words = [word for word in words if word.isalpha() and len(word) > 1]

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_words = [word for word in lemmatized_words if len(word) > 1]

    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text