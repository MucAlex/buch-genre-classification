import pandas as pd
import numpy as np
from multiprocessing import Pool
import nltk
from nltk import word_tokenize

from nltk.stem.cistem import Cistem
from typing import Callable, Optional
import re


class TextProcessor:
    """
    Klasse zum Bereinigen von Texten.

    """

    def __init__(
        self,
        text_data: Optional[pd.DataFrame] = None,
        text_column: str = 'body',
    ):
        self.text_data: pd.DataFrame = text_data
        self.text_column: str = text_column

    def join_list2text(self, data):
        """
        Fügt Tokens, die in Liste gespeichert sind, wieder zu einem String zusammen
        """
        return data[self.text_column].apply(lambda words: " ".join(words)).to_frame()

    def clean_text(self, data, for_embedding=False):
        """
        Bereinigen der Texte. Entfernt einzelne Buchstaben, Zahlen, Symbole und Whitespace.

        """
        RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
        if for_embedding:
            # Keep punctuation
            RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
            RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)
        else:
            RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
            RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)

        data = data[self.text_column].replace(RE_ASCII, " ", regex=True).to_frame()
        data = data[self.text_column].replace(RE_SINGLECHAR, " ", regex=True).to_frame()
        data = data[self.text_column].replace(RE_WSPACE, " ", regex=True).to_frame()
        return data

    def remove_stopwords(self, data):
        """
        Entfernt Stopwörter
        """
        from nltk.corpus import stopwords
        try:
            stopwords = stopwords.words('german')
        except:
            nltk.download('stopwords')
            stopwords = stopwords.words('german')
        stopwords = set(stopwords)
        return data[self.text_column].apply(lambda text: [word for word in text if word not in stopwords]).to_frame()

    def stemmer(self, data):
        """
        Führt stemming für die Wörter in den Listen durch. Nutzt dabei den Ci-Stemmer von Nltk
        """

        ci_stemmer = Cistem()
        return data[self.text_column].apply(lambda text: [ci_stemmer.stem(word) for word in text]).to_frame()

    def tokenizer(self, data):
        """
        Tokenisiert die Texte. Macht aus Texten eine Liste von Tokens (Wörtern)

        """
        return data[self.text_column].apply(word_tokenize, args=("german", )).to_frame()

    def to_lower(self, data):
        """
        Alle Wörter auf Kleinschreibung
        """
        return data[self.text_column].str.lower().to_frame()

    def preprocess(self, data: pd.DataFrame, for_embedding=False):
        """ Führt alle Methoden aus zur Bereinigung der Texte"""
        data = self.clean_text(data, for_embedding=for_embedding)
        data = self.tokenizer(data)
        if not for_embedding:
            data = self.stemmer(data)
            data = self.remove_stopwords(data)
            data = self.join_list2text(data)
        return data


