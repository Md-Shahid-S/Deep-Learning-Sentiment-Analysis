import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Run once: nltk.download(['punkt','stopwords','wordnet'])

class NLTKPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Keep negations — critical for sentiment
        self.stop_words -= {"no", "not", "nor", "never", "neither"}

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)       # strip HTML tags
        text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation/digits
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        tokens = word_tokenize(self.clean(text))
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def process(self, text: str) -> str:
        """Return preprocessed string (for TF-IDF baselines)."""
        return " ".join(self.tokenize(text))