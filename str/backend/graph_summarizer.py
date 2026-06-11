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
  5. Usar heapq (fila de prioridade máxima) para selecionar as N frases
     com maior score → resumo final
"""

import time
import heapq
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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

    def stats(self) -> Dict:
        degrees = {i: len(v) for i, v in self.adj.items()}
        return {
            "vertices": self.n,
            "arestas": self.n_edges,
            "grau_medio": sum(degrees.values()) / max(self.n, 1),
            "grau_max": max(degrees.values()) if degrees else 0,
        }


# ═══════════════════════════════════════════════════════════════
# PAGERANK (implementado do zero)
# ═══════════════════════════════════════════════════════════════

def pagerank(
    graph: SentenceGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[int, float]:
    """
    PageRank adaptado para grafos de frases (TextRank).
    
    Fórmula clássica:
      PR(v) = (1 - d) / N  +  d × Σ [ PR(u) × w(u,v) / Σ_k w(u,k) ]
    
    onde:
      d       = fator de amortecimento (damping)
      N       = total de vértices
      w(u,v)  = peso da aresta u→v
    
    Parâmetros:
      damping  : probabilidade de seguir uma aresta (default 0.85)
      max_iter : máximo de iterações
      tol      : tolerância para convergência (L1)
    """
    n = graph.n
    if n == 0:
        return {}

    # Inicializa scores uniformemente
    scores: Dict[int, float] = {i: 1.0 / n for i in range(n)}

    for iteration in range(max_iter):
        new_scores: Dict[int, float] = {}

        for v in range(n):
            rank_sum = 0.0
            # Contribuição de cada vizinho u que aponta para v
            for u, weight in graph.get_neighbors(v).items():
                total_weight = graph.out_weight_sum(u)
                if total_weight > 0:
                    rank_sum += scores[u] * (weight / total_weight)

            new_scores[v] = (1 - damping) / n + damping * rank_sum

        # Checa convergência (norma L1)
        delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores

        if delta < tol:
            print(f"  PageRank convergiu em {iteration + 1} iterações (δ={delta:.2e})")
            break
    else:
        print(f"  PageRank atingiu máximo de {max_iter} iterações")

    return scores


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
        similarity_method: str = "jaccard",   # "jaccard" | "tfidf"
        similarity_threshold: float = 0.1,
        damping: float = 0.85,
    ):
        self.preprocessor = TextPreprocessor(lang)
        self.similarity_method = similarity_method
        self.threshold = similarity_threshold
        self.damping = damping

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
        auto_threshold: bool = True
    ) -> Dict:
        """
        Sumariza o texto e retorna dict com resumo + metadados do grafo.
        
        Args:
            text:            Texto original (artigo Wikipedia)
            n_sentences:     Número de frases no resumo
            preserve_order:  Manter ordem original das frases no resumo
            auto_threshold:  Calcular threshold automaticamente baseado na similaridade média

        Returns:
            Dict com: resumo, frases selecionadas, scores, stats do grafo, performance
        """
        import time
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

        n_sentences = min(n_sentences, total)

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
        # PASSO 3: Threshold adaptativo (se solicitado)
        # ═══════════════════════════════════════════════════════════════
        original_threshold = self.threshold
        
        if auto_threshold and self.similarity_method == "jaccard" and total > 5:
            print("[3/6] Calculando threshold adaptativo...")
            
            # Coleta amostra de similaridades (limitada para performance)
            sample_sims = []
            sample_limit = min(50, total)  # Amostra dos primeiros 50 ou menos
            
            for i in range(min(sample_limit, total)):
                for j in range(i + 1, min(sample_limit, total)):
                    sim = jaccard_similarity(token_sets[i], token_sets[j])
                    if sim > 0:  # Ignora zeros
                        sample_sims.append(sim)
            
            if sample_sims:
                mean_sim = sum(sample_sims) / len(sample_sims)
                # Calcula desvio padrão
                variance = sum((s - mean_sim) ** 2 for s in sample_sims) / len(sample_sims)
                std_sim = variance ** 0.5
                
                # Threshold = média + 0.3 * desvio (ajustável)
                auto_threshold_value = mean_sim + 0.3 * std_sim
                
                # Limita entre 0.05 e 0.5 para evitar extremos
                auto_threshold_value = max(0.05, min(0.5, auto_threshold_value))
                
                print(f"      ✓ Média das similaridades: {mean_sim:.4f}")
                print(f"      ✓ Desvio padrão: {std_sim:.4f}")
                print(f"      ✓ Threshold original: {self.threshold}")
                print(f"      ✓ Threshold automático: {auto_threshold_value:.4f}")
                
                self.threshold = auto_threshold_value
            else:
                print(f"      ⚠ Sem amostras válidas, mantendo threshold: {self.threshold}")
        else:
            print(f"[3/6] Pulando threshold adaptativo (auto_threshold={auto_threshold}, metodo={self.similarity_method})")

        # ═══════════════════════════════════════════════════════════════
        # PASSO 4: Construção do grafo
        # ═══════════════════════════════════════════════════════════════
        print("[4/6] Calculando similaridade e construindo grafo...")
        graph = SentenceGraph(total)
        similarity_comparisons = 0
        edges_added = 0

        if self.similarity_method == "tfidf":
            tfidf_vecs = compute_tfidf(sentences, self.preprocessor)
            for i in range(total):
                for j in range(i + 1, total):
                    similarity_comparisons += 1
                    sim = cosine_tfidf_similarity(tfidf_vecs[i], tfidf_vecs[j])
                    if sim >= self.threshold:
                        graph.add_edge(i, j, sim)
                        edges_added += 1
        else:  # jaccard (padrão)
            for i in range(total):
                for j in range(i + 1, total):
                    similarity_comparisons += 1
                    sim = jaccard_similarity(token_sets[i], token_sets[j])
                    if sim >= self.threshold:
                        graph.add_edge(i, j, sim)
                        edges_added += 1

        stats = graph.stats()
        print(f"      ✓ Comparações de similaridade: {similarity_comparisons}")
        print(f"      ✓ Arestas adicionadas: {edges_added}")
        print(f"      ✓ Densidade do grafo: {2 * edges_added / (total * (total - 1)) if total > 1 else 0:.4f}")
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
        scores = pagerank(graph, damping=self.damping)
        self._last_scores = scores
        
        # Mostra top 3 scores para debug
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"      ✓ Top scores: {[(idx, round(score, 6)) for idx, score in top_scores]}")

        # ═══════════════════════════════════════════════════════════════
        # PASSO 6: Seleção das frases com fila de prioridade
        # ═══════════════════════════════════════════════════════════════
        print("[6/6] Selecionando frases via fila de prioridade (max-heap)...")
        top = top_k_sentences(scores, sentences, n_sentences)

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
        return {
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
                "metodo_similaridade": self.similarity_method,
                "threshold_original": original_threshold,
                "threshold_usado": self.threshold,
                "threshold_automatico": auto_threshold,
                "grafo": stats,
                "top_tokens": sorted(
                    global_freq.items(), key=lambda x: x[1], reverse=True
                )[:15],
                "performance": {
                    "tempo_segundos": round(elapsed_time, 2),
                    "similaridades_calculadas": similarity_comparisons,
                    "arestas_adicionadas": edges_added,
                    "densidade_grafo": round(2 * edges_added / (total * (total - 1)), 6) if total > 1 else 0,
                    "media_tokens_por_frase": round(avg_tokens_per_sentence, 2),
                    "tokens_unicos": unique_tokens,
                }
            },
        }