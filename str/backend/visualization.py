# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Dict, TYPE_CHECKING

# Importação condicional para evitar circular import
if TYPE_CHECKING:
    from graph_summarizer import SentenceGraph


def visualize_sentence_graph(
    graph,  # type: ignore - evitar erro de tipo
    sentences: List[str],
    scores: Optional[Dict[int, float]] = None,
    output_file: str = "grafo_similaridade.png",
    max_labels: int = 20,
    figsize: tuple = (14, 10)
):
    """
    Visualiza o grafo de similaridade entre frases.
    
    Args:
        graph: SentenceGraph a ser visualizado
        sentences: Lista das frases originais
        scores: Scores PageRank (opcional, para colorir nós)
        output_file: Nome do arquivo de saída
        max_labels: Número máximo de frases a mostrar como label
        figsize: Tamanho da figura
    """
    G = graph.to_networkx()
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Prepara cores baseadas no PageRank
    if scores:
        node_colors = [scores.get(n, 0.0) for n in G.nodes()]
        # Usar cmap diretamente do matplotlib
        cmap = plt.get_cmap('Blues')
    else:
        node_colors = 'lightblue'
        cmap = None
    
    # Desenha nós e arestas
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, 
                           node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, width=0.5)
    
    # Labels apenas para as mais importantes
    if scores and len(sentences) > max_labels:
        # Ordena scores e pega os top indices
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = set(idx for idx, _ in sorted_scores[:max_labels])
    else:
        top_indices = set(range(len(sentences)))
    
    labels = {i: f"{i}: {sentences[i][:40]}..." for i in top_indices if i < len(sentences)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Conta o número de arestas corretamente
    n_edges = graph.n_edges if hasattr(graph, 'n_edges') else G.number_of_edges()
    
    plt.title(f"Grafo de Similaridade | {len(sentences)} frases | {n_edges} arestas")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Fecha a figura para não travar
    print(f"📊 Grafo salvo em: {output_file}")