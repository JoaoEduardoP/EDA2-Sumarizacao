"""
preprocessor.py
---------------
Pré-processamento de texto usando spaCy.
- Tokenização e lematização (lida com tempo verbal, plural/singular)
- Remoção de stopwords
- Segmentação de frases
"""

import re
import spacy
from typing import List, Set


class TextPreprocessor:
    """
    Pré-processa texto usando spaCy com modelos treinados.
    """

    def __init__(self, lang: str = "pt"):
        self.lang = lang
        # Primeiro carrega o modelo
        if lang == "pt":
            self.nlp = spacy.load("pt_core_news_sm")
        else:
            self.nlp = spacy.load("en_core_web_sm")
        # Depois configura as stopwords
        self.stopwords = self.nlp.Defaults.stop_words

    def tokenize(self, text: str) -> List[str]:
        """Tokeniza e lematiza corretamente usando spaCy."""
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            # Filtra pontuação, espaços, stopwords
            if not token.is_punct and not token.is_space and not token.is_stop:
                lemma = token.lemma_
                if len(lemma) > 2:
                    tokens.append(lemma)
        return tokens

    def segment_sentences(self, text: str) -> List[str]:
        """Segmenta texto em frases usando o modelo spaCy."""
        # Remove referências Wikipedia e parênteses
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s.split()) >= 5:  # Filtra frases muito curtas
                sentences.append(s)
        return sentences

    def get_token_set(self, text: str) -> Set[str]:
        """Retorna conjunto de tokens únicos de um texto."""
        return set(self.tokenize(text))