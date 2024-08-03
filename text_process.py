import nltk
import re

# To remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

# For lemmetization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# For tokenization
from nltk.tokenize import word_tokenize
nltk.download('punkt')


stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    print("Raw data", len(text))
    #removing anything other than alphabets
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    #lowercase
    text = text.lower()
    
    words = nltk.word_tokenize(text)
    
    #removing stopword
    words = [i for i in words if i not in stop_words]
    
    # Lemmatization
    #print(len(words))
    new_txt = [lemmatizer.lemmatize(word) for word in words]
    new_txt = " ".join(new_txt)

    print("Processed data", len(new_txt))

    return [new_txt]
