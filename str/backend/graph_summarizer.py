"""
graph_summarizer.py
-------------------
Estruturas de dados principais do projeto:

  Grafo:           Vértice = frase | Aresta = similaridade Jaccard
  Hash table:      frequência de tokens por frase (dict Python)
  Fila de prioridade: heapq para selecionar top-k frases pelo PageRank

Algoritmo:
  1. Tokenizar cada frase → conjunto de tokens lematizados
  2. Calcular similaridade Jaccard entre todos os pares de frases
  3. Adicionar aresta se similaridade > threshold
  4. Rodar PageRank (implementado do zero) sobre o grafo
  5. Usar heapq (fila de prioridade máxima) para obter um pool de frases
     candidatas com maior score
  6. Remover frases redundantes do pool (greedy, baseado em Jaccard) e
     selecionar as N frases finais → resumo final
"""

import time
import heapq
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx  # usado APENAS para visualização do grafo

from preprocessor import TextPreprocessor


# ═══════════════════════════════════════════════════════════════
# SIMILARIDADE
# ═══════════════════════════════════════════════════════════════

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Similaridade de Jaccard: |A ∩ B| / |A ∪ B|
    Mede a sobreposição de tokens entre duas frases.
    Complexidade: O(|A| + |B|)
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def cosine_tfidf_similarity(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float]
) -> float:
    """
    Similaridade de cosseno entre dois vetores TF-IDF.
    cos(θ) = (A · B) / (|A| × |B|)
    """
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compute_summary_sentence_count(
    total_sentences: int,
    n_sentences: Optional[int] = None,
    summary_percent: Optional[float] = None,
) -> int:
    if total_sentences <= 0:
        return 0
    if summary_percent is not None:
        desired = math.ceil(total_sentences * _clamp(float(summary_percent), 0.0, 100.0) / 100)
    else:
        desired = n_sentences if n_sentences is not None else 5
    return max(1, min(int(desired), total_sentences))


def compute_adaptive_threshold(
    similarities: List[float],
    method: str,
    base_threshold: float = 0.1,
    strategy: str = "auto_density",
    target_density: float = 0.12,
) -> Dict[str, Any]:
    total_pairs = len(similarities)
    positive = [s for s in similarities if s > 0]
    mean_sim = sum(similarities) / total_pairs if total_pairs else 0.0
    variance = sum((s - mean_sim) ** 2 for s in similarities) / total_pairs if total_pairs else 0.0
    std_sim = variance ** 0.5

    aliases = {"fixed": "manual", "fixo": "manual", "density": "auto_density", "densidade": "auto_density", "auto": "mean_std", "mean": "mean_std", "mean+std": "mean_std"}
    normalized_strategy = aliases.get((strategy or "auto_density").lower(), (strategy or "auto_density").lower())
    threshold = base_threshold
    desired_edges = 0

    if normalized_strategy == "manual":
        automatic = False
    elif normalized_strategy == "mean_std":
        min_t, max_t, k = {"embeddings": (0.20, 0.85, 0.50), "tfidf": (0.05, 0.75, 0.50)}.get(method, (0.05, 0.50, 0.30))
        source = positive or similarities
        if source:
            local_mean = sum(source) / len(source)
            local_variance = sum((s - local_mean) ** 2 for s in source) / len(source)
            threshold = _clamp(local_mean + k * (local_variance ** 0.5), min_t, max_t)
        automatic = True
    elif normalized_strategy == "auto_density":
        target_density = _clamp(float(target_density), 0.01, 1.0)
        desired_edges = math.ceil(total_pairs * target_density) if total_pairs else 0
        if similarities and desired_edges > 0:
            sorted_sims = sorted(similarities, reverse=True)
            cutoff_index = min(desired_edges, len(sorted_sims)) - 1
            threshold = sorted_sims[cutoff_index]
        automatic = True
    else:
        normalized_strategy = "manual"
        automatic = False

    return {
        "threshold": float(threshold),
        "strategy": normalized_strategy,
        "automatico": automatic,
        "target_density": round(float(target_density), 4),
        "desired_edges": desired_edges,
        "total_pairs": total_pairs,
        "positive_pairs": len(positive),
        "mean_similarity": round(mean_sim, 6),
        "std_similarity": round(std_sim, 6),
    }


# ═══════════════════════════════════════════════════════════════
# TF-IDF (implementação própria, sem sklearn)
# ═══════════════════════════════════════════════════════════════

def compute_tfidf(
    sentences: List[str],
    preprocessor: TextPreprocessor
) -> List[Dict[str, float]]:
    """
    Computa vetores TF-IDF para cada frase.
    
    Estrutura interna:
      freq_table: dict[frase_idx][token] = contagem  ← hash table
    
    TF(t,d)  = freq(t,d) / max_freq(d)
    IDF(t)   = log(N / df(t))   onde df = nº de frases com o token
    TF-IDF   = TF × IDF
    """
    N = len(sentences)
    tokenized = [preprocessor.tokenize(s) for s in sentences]

    # Hash table: frequência de tokens por frase
    freq_table: List[Dict[str, int]] = []
    for tokens in tokenized:
        freq: Dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        freq_table.append(dict(freq))

    # Document frequency
    df: Dict[str, int] = defaultdict(int)
    for freq in freq_table:
        for token in freq:
            df[token] += 1

    # TF-IDF vetores
    tfidf_vectors: List[Dict[str, float]] = []
    for freq in freq_table:
        if not freq:
            tfidf_vectors.append({})
            continue
        max_freq = max(freq.values())
        vec: Dict[str, float] = {}
        for token, count in freq.items():
            tf = count / max_freq
            idf = math.log(N / (df[token] + 1)) + 1
            vec[token] = tf * idf
        tfidf_vectors.append(vec)

    return tfidf_vectors


# ═══════════════════════════════════════════════════════════════
# GRAFO DE SENTENÇAS
# ═══════════════════════════════════════════════════════════════

class SentenceGraph:
    """
    Grafo não-direcionado e ponderado onde:
      - Vértices (nós): índices das frases
      - Arestas: frases com similaridade > threshold
      - Peso: valor da similaridade (Jaccard ou TF-IDF cosseno)
    
    Representação interna: lista de adjacência (dict de dicts)
    """

    def __init__(self, n_sentences: int):
        self.n = n_sentences
        # Lista de adjacência: adj[i][j] = peso da aresta
        self.adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n_sentences)}
        self.n_edges = 0

    def add_edge(self, i: int, j: int, weight: float):
        """Adiciona aresta bidirecional com peso."""
        if i != j and weight > 0:
            self.adj[i][j] = weight
            self.adj[j][i] = weight
            self.n_edges += 1

    def get_neighbors(self, i: int) -> Dict[int, float]:
        return self.adj[i]

    def out_weight_sum(self, i: int) -> float:
        """Soma dos pesos das arestas saindo de i (para PageRank)."""
        return sum(self.adj[i].values())

    def to_networkx(self) -> nx.Graph:
        """Converte para NetworkX (somente para visualização/debug)."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in self.adj:
            for j, w in self.adj[i].items():
                if i < j:
                    G.add_edge(i, j, weight=w)
        return G

    def edge_list(self) -> List[Dict[str, float]]:
        return [
            {"origem": i, "destino": j, "peso": round(w, 6)}
            for i, neighbors in self.adj.items()
            for j, w in neighbors.items()
            if i < j
        ]

    def _component_sizes(self) -> List[int]:
        visited: Set[int] = set()
        sizes: List[int] = []
        for start in range(self.n):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            sizes.append(size)
        return sizes

    def stats(self) -> Dict:
        degrees = {i: len(v) for i, v in self.adj.items()}
        component_sizes = self._component_sizes()
        density = 2 * self.n_edges / (self.n * (self.n - 1)) if self.n > 1 else 0.0
        return {
            "vertices": self.n,
            "arestas": self.n_edges,
            "grau_medio": sum(degrees.values()) / max(self.n, 1),
            "grau_max": max(degrees.values()) if degrees else 0,
            "densidade": round(density, 6),
            "nos_isolados": sum(1 for degree in degrees.values() if degree == 0),
            "componentes_conectados": len(component_sizes),
            "maior_componente": max(component_sizes) if component_sizes else 0,
        }


# ═══════════════════════════════════════════════════════════════
# PAGERANK (implementado do zero)
# ═══════════════════════════════════════════════════════════════

def pagerank(
    graph: SentenceGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_metadata: bool = False,
):
    """
    PageRank adaptado para grafos de frases (TextRank).

    Fórmula clássica (com tratamento de dangling nodes):
      PR(v) = (1 - d) / N  +  d × [ Σ PR(u)·w(u,v)/Σ_k w(u,k)  +  dangling_mass / N ]

    onde:
      d              = fator de amortecimento (damping)
      N              = total de vértices
      w(u,v)         = peso da aresta u→v
      dangling_mass  = soma do PR de vértices SEM arestas de saída
                       (redistribuída uniformemente para conservar a massa
                       total de rank; sem isso, vértices sem vizinhos
                       "vazam" probabilidade do sistema a cada iteração,
                       distorcendo os scores de todos os outros nós)

    Parâmetros:
      damping  : probabilidade de seguir uma aresta (default 0.85)
      max_iter : máximo de iterações
      tol      : tolerância para convergência (L1)
    """
    n = graph.n
    if n == 0:
        scores: Dict[int, float] = {}
        metadata = {
            "convergiu": True,
            "iteracoes": 0,
            "delta_final": 0.0,
            "soma_scores": 0.0,
            "damping": damping,
            "max_iter": max_iter,
            "tol": tol,
            "dangling_nodes": 0,
        }
        return (scores, metadata) if return_metadata else scores

    # Inicializa scores uniformemente
    scores: Dict[int, float] = {i: 1.0 / n for i in range(n)}

    # Vértices sem arestas de saída (dangling nodes). A estrutura do grafo
    # não muda entre iterações, então essa lista é calculada uma única vez
    # fora do loop principal.
    dangling_nodes = [i for i in range(n) if graph.out_weight_sum(i) == 0]
    iterations_run = 0
    final_delta = 0.0
    converged = False

    for iteration in range(max_iter):
        new_scores: Dict[int, float] = {}

        # Massa de rank presa em dangling nodes nesta iteração, redistribuída
        # igualmente entre todos os vértices (tratamento padrão de dangling
        # nodes em PageRank).
        dangling_mass = sum(scores[i] for i in dangling_nodes)
        dangling_contribution = dangling_mass / n

        for v in range(n):
            rank_sum = 0.0
            # Contribuição de cada vizinho u que aponta para v
            for u, weight in graph.get_neighbors(v).items():
                total_weight = graph.out_weight_sum(u)
                if total_weight > 0:
                    rank_sum += scores[u] * (weight / total_weight)

            # Redistribui a massa dos dangling nodes para v
            rank_sum += dangling_contribution

            new_scores[v] = (1 - damping) / n + damping * rank_sum

        # Checa convergência (norma L1)
        delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        iterations_run = iteration + 1
        final_delta = delta

        if delta < tol:
            converged = True
            print(f"  PageRank convergiu em {iteration + 1} iterações (δ={delta:.2e})")
            break
    else:
        print(f"  PageRank atingiu máximo de {max_iter} iterações")

    metadata = {
        "convergiu": converged,
        "iteracoes": iterations_run,
        "delta_final": round(final_delta, 12),
        "soma_scores": round(sum(scores.values()), 12),
        "damping": damping,
        "max_iter": max_iter,
        "tol": tol,
        "dangling_nodes": len(dangling_nodes),
    }

    return (scores, metadata) if return_metadata else scores


# ═══════════════════════════════════════════════════════════════
# FILA DE PRIORIDADE (max-heap via heapq)
# ═══════════════════════════════════════════════════════════════

def top_k_sentences(
    scores: Dict[int, float],
    sentences: List[str],
    k: int
) -> List[Tuple[float, int, str]]:
    """
    Seleciona as k frases com maior score usando max-heap.
    Complexidade: O(n log k)
    """
    if k <= 0:
        return []
    
    heap = []  # min-heap de (score, idx) - guarda os k maiores
    
    for idx, score in scores.items():
        if len(heap) < k:
            heapq.heappush(heap, (score, idx))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, idx))
    
    # Ordena do maior para o menor
    result = [(score, idx, sentences[idx]) for score, idx in sorted(heap, reverse=True)]
    return result


# ═══════════════════════════════════════════════════════════════
# REMOÇÃO DE REDUNDÂNCIA
# ═══════════════════════════════════════════════════════════════

def remove_redundant(
    candidates: List[Tuple[float, int, str]],
    token_sets: List[Set[str]],
    n_sentences: int,
    redundancy_threshold: float = 0.75,
) -> Tuple[List[Tuple[float, int, str]], int]:
    """
    Filtra frases redundantes de um conjunto de candidatas já ranqueadas.

    Estratégia gulosa (greedy): percorre as candidatas em ordem decrescente
    de score PageRank e só aceita uma frase no resumo se ela NÃO for muito
    parecida (Jaccard > redundancy_threshold) com nenhuma frase já aceita.
    Isso evita que o resumo final tenha duas frases dizendo essencialmente
    a mesma coisa, mesmo que ambas tenham scores altos por estarem na
    mesma "vizinhança" densa do grafo.

    Args:
        candidates:            Lista (score, índice, frase) já ordenada
                                por score decrescente (pool maior que o
                                tamanho final do resumo).
        token_sets:            Conjuntos de tokens de cada frase (reaproveitados
                                do pré-processamento, índice = índice da frase).
        n_sentences:           Quantidade final de frases desejada no resumo.
        redundancy_threshold:  Acima desse valor de Jaccard, a frase candidata
                                é considerada redundante e descartada.

    Returns:
        (frases_selecionadas, quantidade_removida_por_redundancia)
    """
    selected: List[Tuple[float, int, str]] = []
    selected_idx: List[int] = []
    removed_count = 0

    for score, idx, sentence in candidates:
        if len(selected) >= n_sentences:
            break

        is_redundant = any(
            jaccard_similarity(token_sets[idx], token_sets[sel_idx]) > redundancy_threshold
            for sel_idx in selected_idx
        )

        if is_redundant:
            removed_count += 1
            continue

        selected.append((score, idx, sentence))
        selected_idx.append(idx)

    # Fallback: se o filtro foi agressivo e a pool não foi suficiente para
    # completar n_sentences, completa com as melhores frases restantes,
    # ignorando o filtro de redundância (preferimos um resumo completo a
    # um resumo curto demais).
    if len(selected) < n_sentences:
        for score, idx, sentence in candidates:
            if len(selected) >= n_sentences:
                break
            if idx not in selected_idx:
                selected.append((score, idx, sentence))
                selected_idx.append(idx)

    return selected, removed_count


def select_sentences_mmr(
    candidates: List[Tuple[float, int, str]],
    token_sets: List[Set[str]],
    n_sentences: int,
    diversity_alpha: float = 0.85,
) -> List[Tuple[float, int, str]]:
    if n_sentences <= 0 or not candidates:
        return []
    alpha = _clamp(float(diversity_alpha), 0.0, 1.0)
    max_score = max(score for score, _, _ in candidates) or 1.0
    remaining = {idx: (score, idx, sentence) for score, idx, sentence in candidates}
    selected: List[Tuple[float, int, str]] = []
    selected_idx: List[int] = []
    while remaining and len(selected) < n_sentences:
        best_item: Optional[Tuple[float, int, str]] = None
        best_mmr = float("-inf")
        for score, idx, sentence in remaining.values():
            importance = score / max_score
            redundancy = max((jaccard_similarity(token_sets[idx], token_sets[s]) for s in selected_idx), default=0.0)
            mmr_score = alpha * importance - (1 - alpha) * redundancy
            if mmr_score > best_mmr or (math.isclose(mmr_score, best_mmr) and best_item and score > best_item[0]):
                best_mmr = mmr_score
                best_item = (score, idx, sentence)
        if best_item is None:
            break
        selected.append(best_item)
        selected_idx.append(best_item[1])
        del remaining[best_item[1]]
    return selected


# ═══════════════════════════════════════════════════════════════
# SUMARIZADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════

class WikiSummarizer:
    """
    Pipeline completo de sumarização extrativa baseada em grafos.
    
    Pipeline:
      texto → segmentar frases → tokenizar → TF-IDF ou Jaccard
            → construir grafo → PageRank → fila de prioridade → resumo
    """

    def __init__(
        self,
        lang: str = "pt",
        similarity_method: str = "embeddings",  # "jaccard" | "tfidf" | "embeddings"
        similarity_threshold: float = 0.1,
        damping: float = 0.85,
        redundancy_threshold: float = 0.75,
        pagerank_max_iter: int = 100,
        pagerank_tol: float = 1e-6,
    ):
        self.preprocessor = TextPreprocessor(lang)
        self.similarity_method = similarity_method
        self.threshold = similarity_threshold
        self.damping = damping
        self.pagerank_max_iter = pagerank_max_iter
        self.pagerank_tol = pagerank_tol
        # Acima desse valor de Jaccard entre duas frases, a segunda é
        # considerada redundante e não entra no resumo (ver remove_redundant).
        self.redundancy_threshold = redundancy_threshold

    def visualize(self, result: Dict, output_file: str = "grafo.png"):
        """Visualiza o grafo da última sumarização."""
        if hasattr(self, '_last_graph') and hasattr(self, '_last_sentences'):
            from visualization import visualize_sentence_graph
            visualize_sentence_graph(
                self._last_graph,
                self._last_sentences,
                scores=result.get("scores"),
                output_file=output_file
            )

    def summarize(
        self, 
        text: str, 
        n_sentences: int = 5, 
        preserve_order: bool = True, 
        auto_threshold: bool = True,
        summary_percent: Optional[float] = None,
        threshold_strategy: Optional[str] = None,
        target_density: float = 0.12,
        selection_strategy: str = "mmr",
        diversity_alpha: float = 0.85,
        include_graph: bool = False,
    ) -> Dict:
        """
        Sumariza o texto e retorna dict com resumo + metadados do grafo.
        
        Args:
            text:            Texto original (artigo Wikipedia)
            n_sentences:     Número de frases no resumo (ignorado se summary_percent for informado)
            preserve_order:  Manter ordem original das frases no resumo
            auto_threshold:  Compatibilidade: usa threshold automático se threshold_strategy não for informada
            summary_percent: Porcentagem do texto original a manter no resumo
            threshold_strategy: manual | mean_std | auto_density
            target_density: Fração aproximada dos pares que viram arestas em auto_density
            selection_strategy: mmr | redundancy
            diversity_alpha: Peso do PageRank no MMR

        Returns:
            Dict com: resumo, frases selecionadas, scores, stats do grafo, performance
        """
        start_time = time.time()
        
        print("\n" + "═" * 60)
        print("  WIKISUMMARIZER - Iniciando sumarização")
        print("═" * 60)
        
        # ═══════════════════════════════════════════════════════════════
        # PASSO 1: Segmentação de frases
        # ═══════════════════════════════════════════════════════════════
        print("\n[1/6] Segmentando frases...")
        sentences = self.preprocessor.segment_sentences(text)
        total = len(sentences)
        print(f"      ✓ {total} frases encontradas")

        if total == 0:
            return {"erro": "Nenhuma frase encontrada no texto."}

        n_sentences = compute_summary_sentence_count(
            total,
            n_sentences=n_sentences,
            summary_percent=summary_percent,
        )

        # ═══════════════════════════════════════════════════════════════
        # PASSO 2: Tokenização e hash tables
        # ═══════════════════════════════════════════════════════════════
        print("[2/6] Tokenizando e construindo hash tables de frequência...")
        token_sets = [self.preprocessor.get_token_set(s) for s in sentences]

        # Hash table: frequência global de tokens (para debug / análise)
        global_freq: Dict[str, int] = defaultdict(int)
        for ts in token_sets:
            for token in ts:
                global_freq[token] += 1
        
        # Estatísticas de tokens
        unique_tokens = len(global_freq)
        avg_tokens_per_sentence = sum(len(ts) for ts in token_sets) / max(total, 1)
        print(f"      ✓ Tokens únicos: {unique_tokens}")
        print(f"      ✓ Média de tokens por frase: {avg_tokens_per_sentence:.1f}")

        # ═══════════════════════════════════════════════════════════════
        # PASSO 3: Similaridades + threshold adaptativo
        # ═══════════════════════════════════════════════════════════════
        original_threshold = self.threshold
        normalized_threshold_strategy = (
            threshold_strategy
            if threshold_strategy is not None
            else ("auto_density" if auto_threshold else "manual")
        )

        print("[3/6] Calculando similaridades e threshold...")
        similarity_pairs: List[Tuple[int, int, float]] = []

        if self.similarity_method == "embeddings":
            print("      → Gerando embeddings semânticos...")
            from embedder import get_embeddings, cosine_similarity_matrix
            embeddings = get_embeddings(sentences)
            sim_matrix = cosine_similarity_matrix(embeddings)
            print(f"      ✓ Embeddings gerados: {embeddings.shape}")

            for i in range(total):
                for j in range(i + 1, total):
                    similarity_pairs.append((i, j, float(sim_matrix[i, j])))
        elif self.similarity_method == "tfidf":
            tfidf_vecs = compute_tfidf(sentences, self.preprocessor)
            for i in range(total):
                for j in range(i + 1, total):
                    sim = cosine_tfidf_similarity(tfidf_vecs[i], tfidf_vecs[j])
                    similarity_pairs.append((i, j, sim))
        else:
            for i in range(total):
                for j in range(i + 1, total):
                    sim = jaccard_similarity(token_sets[i], token_sets[j])
                    similarity_pairs.append((i, j, sim))

        threshold_info = compute_adaptive_threshold(
            [sim for _, _, sim in similarity_pairs],
            method=self.similarity_method,
            base_threshold=original_threshold,
            strategy=normalized_threshold_strategy,
            target_density=target_density,
        )
        self.threshold = threshold_info["threshold"]

        print(f"      ✓ Comparações de similaridade: {len(similarity_pairs)}")
        print(f"      ✓ Estratégia de threshold: {threshold_info['strategy']}")
        print(f"      ✓ Média das similaridades: {threshold_info['mean_similarity']:.4f}")
        print(f"      ✓ Desvio padrão: {threshold_info['std_similarity']:.4f}")
        print(f"      ✓ Threshold original: {original_threshold}")
        print(f"      ✓ Threshold usado: {self.threshold:.4f}")

        # ═══════════════════════════════════════════════════════════════
        # PASSO 4: Construção do grafo
        # ═══════════════════════════════════════════════════════════════
        print("[4/6] Construindo grafo...")
        graph = SentenceGraph(total)
        edges_added = 0

        for i, j, sim in similarity_pairs:
            if sim >= self.threshold:
                before = graph.n_edges
                graph.add_edge(i, j, sim)
                if graph.n_edges > before:
                    edges_added += 1

        stats = graph.stats()
        print(f"      ✓ Arestas adicionadas: {edges_added}")
        print(f"      ✓ Densidade do grafo: {stats['densidade']:.4f}")
        print(f"      ✓ Grau médio: {stats['grau_medio']:.2f}")
        print(f"      ✓ Grau máximo: {stats['grau_max']}")

        # Armazena grafo e frases para visualização posterior
        self._last_graph = graph
        self._last_sentences = sentences
        self._last_scores = None

        # ═══════════════════════════════════════════════════════════════
        # PASSO 5: PageRank
        # ═══════════════════════════════════════════════════════════════
        print("[5/6] Executando PageRank...")
        scores, pagerank_info = pagerank(
            graph,
            damping=self.damping,
            max_iter=self.pagerank_max_iter,
            tol=self.pagerank_tol,
            return_metadata=True,
        )
        self._last_scores = scores
        
        # Mostra top 3 scores para debug
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"      ✓ Top scores: {[(idx, round(score, 6)) for idx, score in top_scores]}")
        print(
            "      ✓ Convergência: "
            f"{pagerank_info['convergiu']} | "
            f"{pagerank_info['iteracoes']} iterações | "
            f"delta={pagerank_info['delta_final']:.2e}"
        )

        # ═══════════════════════════════════════════════════════════════
        # PASSO 6: Seleção das frases
        # ═══════════════════════════════════════════════════════════════
        print("[6/6] Selecionando frases...")

        # Pega um pool maior que n_sentences, pois a etapa de remoção de
        # redundância pode descartar algumas candidatas e precisamos de
        # margem para completar o resumo com frases ainda relevantes.
        pool_size = min(total, max(n_sentences * 3, n_sentences + 10))
        candidates_pool = top_k_sentences(scores, sentences, pool_size)

        print(f"      ✓ Pool de candidatas: {pool_size} frases")
        normalized_selection_strategy = (selection_strategy or "mmr").lower()
        if normalized_selection_strategy in {"pagerank", "redundancy", "baseline"}:
            normalized_selection_strategy = "redundancy"
            print("      → Aplicando baseline PageRank + filtro de redundância "
                  f"(Jaccard > {self.redundancy_threshold})...")
            top, n_redundant_removed = remove_redundant(
                candidates_pool,
                token_sets,
                n_sentences,
                redundancy_threshold=self.redundancy_threshold,
            )
            print(f"      ✓ Frases redundantes descartadas: {n_redundant_removed}")
        else:
            normalized_selection_strategy = "mmr"
            print(f"      → Aplicando MMR (alpha={diversity_alpha})...")
            top = select_sentences_mmr(
                candidates_pool,
                token_sets,
                n_sentences,
                diversity_alpha=diversity_alpha,
            )
            n_redundant_removed = 0

        if preserve_order:
            # Reordena pelo índice original para manter coesão
            top_sorted = sorted(top, key=lambda x: x[1])
        else:
            top_sorted = top

        summary_sentences = [item[2] for item in top_sorted]
        summary = " ".join(summary_sentences)

        # ═══════════════════════════════════════════════════════════════
        # TIMER E PERFORMANCE
        # ═══════════════════════════════════════════════════════════════
        elapsed_time = time.time() - start_time
        
        print("\n" + "─" * 60)
        print(f"⏱️  Tempo total: {elapsed_time:.2f} segundos")
        print(f"📊 Resumo gerado com {n_sentences} frases")
        print("═" * 60 + "\n")

        # ═══════════════════════════════════════════════════════════════
        # RESULTADO
        # ═══════════════════════════════════════════════════════════════
        result = {
            "resumo": summary,
            "frases_selecionadas": [
                {
                    "indice": item[1],
                    "score_pagerank": round(item[0], 6),
                    "frase": item[2],
                }
                for item in top_sorted
            ],
            "scores": scores,  # Adicionado para visualização
            "metadados": {
                "total_frases": total,
                "frases_no_resumo": n_sentences,
                "summary_percent": summary_percent,
                "metodo_similaridade": self.similarity_method,
                "threshold_original": original_threshold,
                "threshold_usado": self.threshold,
                "threshold_automatico": threshold_info["automatico"],
                "threshold": threshold_info,
                "selection_strategy": normalized_selection_strategy,
                "diversity_alpha": round(float(diversity_alpha), 4),
                "redundancy_threshold": self.redundancy_threshold,
                "frases_redundantes_removidas": n_redundant_removed,
                "pagerank": pagerank_info,
                "grafo": stats,
                "top_tokens": sorted(
                    global_freq.items(), key=lambda x: x[1], reverse=True
                )[:15],
                "performance": {
                    "tempo_segundos": round(elapsed_time, 2),
                    "similaridades_calculadas": len(similarity_pairs),
                    "arestas_adicionadas": edges_added,
                    "densidade_grafo": stats["densidade"],
                    "media_tokens_por_frase": round(avg_tokens_per_sentence, 2),
                    "tokens_unicos": unique_tokens,
                }
            },
        }

        if include_graph:
            result["grafo_serializado"] = {
                "arestas": graph.edge_list(),
            }

        return result
