"""
embedder.py
-----------
Geração de embeddings semânticos com sentence-transformers e
cálculo de similaridade por cosseno implementado do zero com NumPy.

O modelo 'paraphrase-multilingual-MiniLM-L12-v2' (~120 MB) suporta
50+ idiomas, incluindo português, e é baixado automaticamente na
primeira execução.
"""

from __future__ import annotations

import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

# Modelo em cache para evitar recarregar a cada chamada
_MODEL_CACHE: dict[str, SentenceTransformer] = {}
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Carrega o modelo e guarda em cache (hash table em memória)."""
    if model_name not in _MODEL_CACHE:
        print(f"  [embedder] Carregando modelo '{model_name}'...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        print(f"  [embedder] Modelo carregado.")
    return _MODEL_CACHE[model_name]


def get_embeddings(
    sentences: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Gera embeddings para uma lista de frases.

    Args:
        sentences:  Lista de frases.
        model_name: Nome do modelo sentence-transformers.
        batch_size: Tamanho do lote para inferência.

    Returns:
        Matriz numpy (n_frases × dim_embedding).
    """
    model = load_model(model_name)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings  # shape: (n, d)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de similaridade por cosseno entre todos os pares.

    Implementação própria (sem sklearn):
        cos(a, b) = (a · b) / (‖a‖ · ‖b‖)

    Realizado em forma matricial:
        1. Normaliza cada vetor (norma L2)
        2. Produto interno da matriz normalizada por sua transposta

    Complexidade: O(n² · d)  onde d = dimensão do embedding

    Args:
        embeddings: Matriz (n × d).

    Returns:
        Matriz simétrica (n × n) com valores em [-1, 1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Evita divisão por zero
    norms = np.where(norms < 1e-9, 1e-9, norms)
    normalized = embeddings / norms          # cada linha tem norma 1
    sim_matrix = normalized @ normalized.T   # produto escalar = cosseno
    # Clipa para [-1, 1] para evitar erros numéricos
    return np.clip(sim_matrix, -1.0, 1.0)


def cosine_similarity_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Similaridade por cosseno entre dois vetores individuais."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
