"""
test_wikisummarizer.py
-----------------------------
Testes unitários do WikiSummarizer.

Execute com:
    cd src
    python -m unittest test.test_wikisummarizer -v
ou:
    cd src
    python test/test_wikisummarizer.py
"""

import sys
import os
import math
import unittest

# Adiciona o caminho para a pasta backend (um nível acima)
# __file__ é o caminho do arquivo de teste (src/test/test_wikisummarizer.py)
# os.path.dirname(__file__) vai para src/test/
# os.path.dirname(...) vai para src/
# então juntamos com 'backend'
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
sys.path.insert(0, backend_path)

# Agora os imports funcionam
from preprocessor import TextPreprocessor
from graph_summarizer import (
    jaccard_similarity,
    cosine_tfidf_similarity,
    compute_adaptive_threshold,
    compute_summary_sentence_count,
    compute_tfidf,
    SentenceGraph,
    pagerank,
    select_sentences_mmr,
    top_k_sentences,
    WikiSummarizer,
)


# ═══════════════════════════════════════════════════════════════
# 1. SIMILARIDADE DE JACCARD
# ═══════════════════════════════════════════════════════════════

class TestJaccardSimilarity(unittest.TestCase):

    def test_identicos(self):
        """Conjuntos idênticos → similaridade 1.0"""
        a = {"grafo", "vertice", "aresta"}
        self.assertAlmostEqual(jaccard_similarity(a, a), 1.0)

    def test_disjuntos(self):
        """Conjuntos sem interseção → similaridade 0.0"""
        a = {"grafo", "vertice"}
        b = {"inteligencia", "artificial"}
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.0)

    def test_parcial(self):
        """Interseção parcial: |A∩B|/|A∪B| = 1/5"""
        a = {"grafo", "vertice", "aresta"}
        b = {"vertice", "peso", "caminho"}
        resultado = jaccard_similarity(a, b)
        self.assertAlmostEqual(resultado, 1 / 5)

    def test_conjunto_vazio(self):
        """Conjunto vazio → similaridade 0.0"""
        self.assertAlmostEqual(jaccard_similarity(set(), {"grafo"}), 0.0)
        self.assertAlmostEqual(jaccard_similarity(set(), set()), 0.0)

    def test_simetria(self):
        """Jaccard deve ser simétrica"""
        a = {"a", "b", "c"}
        b = {"b", "c", "d", "e"}
        self.assertAlmostEqual(jaccard_similarity(a, b), jaccard_similarity(b, a))


# ═══════════════════════════════════════════════════════════════
# 2. SIMILARIDADE COSSENO (TF-IDF)
# ═══════════════════════════════════════════════════════════════

class TestCosineSimilarity(unittest.TestCase):

    def test_vetores_identicos(self):
        """Vetores iguais → cosseno 1.0"""
        v = {"grafo": 0.5, "aresta": 0.3}
        self.assertAlmostEqual(cosine_tfidf_similarity(v, v), 1.0)

    def test_vetores_ortogonais(self):
        """Sem tokens em comum → cosseno 0.0"""
        a = {"grafo": 1.0}
        b = {"inteligencia": 1.0}
        self.assertAlmostEqual(cosine_tfidf_similarity(a, b), 0.0)

    def test_vetor_nulo(self):
        """Vetor nulo → cosseno 0.0"""
        a = {}
        b = {"grafo": 1.0}
        self.assertAlmostEqual(cosine_tfidf_similarity(a, b), 0.0)

    def test_proporcional(self):
        """Vetores proporcionais têm cosseno 1.0"""
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 2.0, "y": 4.0}
        self.assertAlmostEqual(cosine_tfidf_similarity(a, b), 1.0, places=5)

    def test_calculo_manual(self):
        """Cálculo manual: a=(1,0), b=(1,1) → cos=1/sqrt(2)"""
        a = {"x": 1.0}
        b = {"x": 1.0, "y": 1.0}
        esperado = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(cosine_tfidf_similarity(a, b), esperado, places=5)


# ═══════════════════════════════════════════════════════════════
# 3. GRAFO DE SENTENÇAS
# ═══════════════════════════════════════════════════════════════

class TestSentenceGraph(unittest.TestCase):

    def test_adicionar_arestas(self):
        """Arestas são adicionadas corretamente e são bidirecionais"""
        g = SentenceGraph(4)
        g.add_edge(0, 1, 0.5)
        g.add_edge(1, 2, 0.3)

        self.assertIn(1, g.adj[0])
        self.assertIn(0, g.adj[1])
        self.assertAlmostEqual(g.adj[0][1], 0.5)
        self.assertEqual(g.n_edges, 2)

    def test_self_loop_ignorado(self):
        """Arestas de um nó para si mesmo não devem ser adicionadas"""
        g = SentenceGraph(3)
        g.add_edge(0, 0, 1.0)
        self.assertEqual(g.n_edges, 0)
        self.assertNotIn(0, g.adj[0])

    def test_out_weight_sum(self):
        """Soma dos pesos de saída calculada corretamente"""
        g = SentenceGraph(3)
        g.add_edge(0, 1, 0.4)
        g.add_edge(0, 2, 0.6)
        self.assertAlmostEqual(g.out_weight_sum(0), 1.0)

    def test_stats(self):
        """Estatísticas do grafo"""
        g = SentenceGraph(3)
        g.add_edge(0, 1, 0.5)
        g.add_edge(1, 2, 0.3)
        stats = g.stats()
        self.assertEqual(stats["vertices"], 3)
        self.assertEqual(stats["arestas"], 2)
        self.assertAlmostEqual(stats["grau_medio"], 4 / 3)
        self.assertEqual(stats["grau_max"], 2)

    def test_grafo_vazio(self):
        """Grafo sem arestas não causa erro"""
        g = SentenceGraph(5)
        stats = g.stats()
        self.assertEqual(stats["arestas"], 0)
        self.assertEqual(stats["grau_medio"], 0.0)


# ═══════════════════════════════════════════════════════════════
# 4. PAGERANK
# ═══════════════════════════════════════════════════════════════

class TestPageRank(unittest.TestCase):

    def test_grafo_vazio(self):
        """PageRank em grafo sem arestas retorna scores uniformes"""
        g = SentenceGraph(3)
        scores = pagerank(g)
        for i in range(3):
            self.assertAlmostEqual(scores[i], 1 / 3, places=4)

    def test_scores_somam_um(self):
        """Soma dos scores PageRank deve convergir para ~1.0"""
        g = SentenceGraph(4)
        g.add_edge(0, 1, 0.5)
        g.add_edge(1, 2, 0.4)
        g.add_edge(2, 3, 0.3)
        g.add_edge(0, 3, 0.2)
        scores = pagerank(g)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=3)

    def test_no_mais_conectado_tem_maior_score(self):
        """Nó com mais conexões deve ter maior PageRank"""
        g = SentenceGraph(4)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(0, 3, 0.5)
        scores = pagerank(g)
        self.assertEqual(max(scores, key=scores.get), 0)

    def test_grafo_n_zero(self):
        """PageRank em grafo sem nós retorna dict vazio"""
        g = SentenceGraph(0)
        scores = pagerank(g)
        self.assertEqual(scores, {})

    def test_convergencia(self):
        """PageRank deve convergir antes do max_iter"""
        g = SentenceGraph(5)
        for i in range(4):
            g.add_edge(i, i + 1, 0.5)
        scores = pagerank(g, max_iter=200, tol=1e-8)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(v >= 0 for v in scores.values()))


# ═══════════════════════════════════════════════════════════════
# 5. FILA DE PRIORIDADE (top_k_sentences)
# ═══════════════════════════════════════════════════════════════

class TestTopKSentences(unittest.TestCase):

    def setUp(self):
        self.sentences = [
            "Frases sobre grafos são interessantes.",
            "Inteligência artificial transforma o mundo.",
            "PageRank ordena nós por importância.",
            "Jaccard mede sobreposição de tokens.",
            "TF-IDF pondera termos por frequência inversa.",
        ]
        self.scores = {0: 0.10, 1: 0.30, 2: 0.25, 3: 0.15, 4: 0.20}

    def test_top_1(self):
        """Deve retornar a frase com maior score"""
        top = top_k_sentences(self.scores, self.sentences, k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][1], 1)

    def test_top_3_indices_corretos(self):
        """Top-3 deve conter os índices de maior score"""
        top = top_k_sentences(self.scores, self.sentences, k=3)
        indices = {item[1] for item in top}
        self.assertEqual(indices, {1, 2, 4})

    def test_k_maior_que_n(self):
        """Se k > n, retorna tudo sem erro"""
        top = top_k_sentences(self.scores, self.sentences, k=10)
        self.assertEqual(len(top), 5)

    def test_scores_decrescentes(self):
        """Scores retornados devem estar em ordem decrescente"""
        top = top_k_sentences(self.scores, self.sentences, k=5)
        scores_list = [item[0] for item in top]
        self.assertEqual(scores_list, sorted(scores_list, reverse=True))


# ═══════════════════════════════════════════════════════════════
# 6. CONTROLES ADAPTATIVOS
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveControls(unittest.TestCase):

    def test_percentual_e_threshold_por_densidade(self):
        self.assertEqual(compute_summary_sentence_count(10, n_sentences=5, summary_percent=25), 3)
        self.assertEqual(compute_summary_sentence_count(4, summary_percent=1), 1)
        self.assertEqual(compute_summary_sentence_count(4, summary_percent=100), 4)
        info = compute_adaptive_threshold(
            [0.90, 0.80, 0.20, 0.10], method="jaccard", strategy="auto_density", target_density=0.50
        )
        self.assertEqual(info["desired_edges"], 2)
        self.assertAlmostEqual(info["threshold"], 0.80)

    def test_pagerank_metadata_e_mmr(self):
        g = SentenceGraph(3)
        g.add_edge(0, 1, 0.5)
        g.add_edge(1, 2, 0.5)
        scores, meta = pagerank(g, return_metadata=True)
        self.assertEqual(len(scores), 3)
        self.assertIn("iteracoes", meta)
        self.assertIn("delta_final", meta)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=3)

        candidates = [
            (0.50, 0, "Grafos usam vértices e arestas."),
            (0.49, 1, "Grafos também usam vértices e arestas."),
            (0.30, 2, "PageRank mede importância de nós."),
        ]
        token_sets = [
            {"grafo", "vertice", "aresta"},
            {"grafo", "vertice", "aresta"},
            {"pagerank", "importancia", "no"},
        ]
        selected = select_sentences_mmr(candidates, token_sets, 2, diversity_alpha=0.50)
        self.assertEqual([item[1] for item in selected], [0, 2])


# ═══════════════════════════════════════════════════════════════
# 7. PRÉ-PROCESSADOR
# ═══════════════════════════════════════════════════════════════

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.pp = TextPreprocessor(lang="pt")

    def test_segmentacao_basica(self):
        """Texto com duas frases deve gerar duas frases segmentadas"""
        texto = "Teoria dos grafos estuda vértices e arestas. O algoritmo de PageRank classifica nós por importância."
        frases = self.pp.segment_sentences(texto)
        self.assertEqual(len(frases), 2)

    def test_remocao_referencias_wikipedia(self):
        """Referências [1], [2] devem ser removidas"""
        texto = "Grafos foram estudados por Euler.[1] A teoria evoluiu muito.[2] Hoje é amplamente aplicada."
        frases = self.pp.segment_sentences(texto)
        for frase in frases:
            self.assertNotIn("[1]", frase)
            self.assertNotIn("[2]", frase)

    def test_filtragem_frases_curtas(self):
        """Frases com menos de 5 palavras devem ser descartadas"""
        texto = "Ok. Muito bem. Inteligência artificial é um campo vasto da ciência da computação moderna."
        frases = self.pp.segment_sentences(texto)
        for frase in frases:
            self.assertGreaterEqual(len(frase.split()), 5)

    def test_token_set(self):
        """get_token_set deve retornar um set (sem duplicatas)"""
        ts = self.pp.get_token_set("grafo grafo vértice aresta aresta")
        self.assertIsInstance(ts, set)


# ═══════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLETO (WikiSummarizer)
# ═══════════════════════════════════════════════════════════════

class TestWikiSummarizer(unittest.TestCase):

    TEXTO_TESTE = """
        Teoria dos grafos é um ramo da matemática que estuda vértices e arestas.
        Um grafo é formado por um conjunto de vértices e arestas que os conectam.
        O algoritmo de Dijkstra encontra o caminho mais curto entre vértices em grafos ponderados.
        A busca em largura percorre o grafo visitando todos os vizinhos de cada vértice.
        PageRank é um algoritmo que avalia a importância de nós em um grafo direcionado.
    """

    def test_resumo_jaccard(self):
        """Pipeline com Jaccard deve retornar resumo sem erros"""
        s = WikiSummarizer(similarity_method="jaccard", similarity_threshold=0.05)
        resultado = s.summarize(self.TEXTO_TESTE, n_sentences=3)
        self.assertIn("resumo", resultado)
        self.assertNotIn("erro", resultado)

    def test_resumo_tfidf(self):
        """Pipeline com TF-IDF deve retornar resumo sem erros"""
        s = WikiSummarizer(similarity_method="tfidf", similarity_threshold=0.05)
        resultado = s.summarize(self.TEXTO_TESTE, n_sentences=3)
        self.assertIn("resumo", resultado)
        self.assertNotIn("erro", resultado)

    def test_numero_de_frases_respeitado(self):
        """Resumo deve ter exatamente n_sentences frases selecionadas"""
        s = WikiSummarizer(similarity_method="jaccard", similarity_threshold=0.05)
        resultado = s.summarize(self.TEXTO_TESTE, n_sentences=3)
        self.assertEqual(resultado["metadados"]["frases_no_resumo"], 3)

    def test_percentual_e_grafo_serializado_no_pipeline(self):
        s = WikiSummarizer(similarity_method="jaccard", similarity_threshold=0.05)
        resultado = s.summarize(
            self.TEXTO_TESTE,
            summary_percent=40,
            threshold_strategy="manual",
            include_graph=True,
        )
        self.assertEqual(resultado["metadados"]["frases_no_resumo"], 2)
        self.assertIn("grafo_serializado", resultado)
        self.assertEqual(
            len(resultado["grafo_serializado"]["arestas"]),
            resultado["metadados"]["grafo"]["arestas"],
        )

    def test_texto_vazio_retorna_erro(self):
        """Texto sem frases válidas deve retornar chave 'erro'"""
        s = WikiSummarizer()
        resultado = s.summarize("", n_sentences=3)
        self.assertIn("erro", resultado)


if __name__ == "__main__":
    unittest.main(verbosity=2)
