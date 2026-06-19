"""
app.py
------
Interface Streamlit do WikiSummarizer.
Execute com: streamlit run app.py
"""

import sys
import re
import io
from urllib.parse import urlparse, unquote

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "backend")

from wiki_fetcher import get_wiki_article, SAMPLE_TEXTS
from graph_summarizer import WikiSummarizer

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="WikiSummarizer",
    page_icon="📚",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

NIVEIS = {
    "Essencial": {
        "frases": 3,
        "descricao": "Captura o núcleo do artigo em 3 frases.",
        "icon": "⚡",
    },
    "Estudo Rápido": {
        "frases": 5,
        "descricao": "Ideal para revisão em poucos minutos.",
        "icon": "📖",
    },
    "Revisão Detalhada": {
        "frases": 9,
        "descricao": "Cobre os pontos principais com mais profundidade.",
        "icon": "🔬",
    },
}

ARTIGOS_EXEMPLO = {
    "Sistema Solar": "Sistema Solar",
    "Inteligência Artificial": "Inteligência artificial",
    "Grafos (matemática)": "Teoria dos grafos",
    "Revolução Industrial": "Revolução Industrial",
    "Biomas do Brasil": "Biomas do Brasil",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extrair_titulo_da_url(url: str) -> str | None:
    """
    Extrai o título do artigo de uma URL da Wikipedia.
    Ex: https://pt.wikipedia.org/wiki/Inteligência_artificial → 'Inteligência artificial'
    """
    try:
        path = urlparse(url).path          # /wiki/Inteligência_artificial
        titulo = path.split("/wiki/")[-1]  # Inteligência_artificial
        titulo = unquote(titulo)           # decodifica %C3%A7 etc.
        return titulo.replace("_", " ")
    except Exception:
        return None


def detectar_lang_da_url(url: str) -> str:
    """Detecta idioma pela URL (pt.wikipedia.org → 'pt')."""
    try:
        host = urlparse(url).hostname  # pt.wikipedia.org
        return host.split(".")[0]
    except Exception:
        return "pt"


@st.cache_data(show_spinner=False)
def buscar_artigo(titulo: str, lang: str) -> str | None:
    return get_wiki_article(titulo, lang=lang)


@st.cache_data(show_spinner=False)
def sumarizar(texto: str, n_frases: int, lang: str, metodo: str) -> dict:
    summarizer = WikiSummarizer(
        lang=lang,
        similarity_method=metodo,
        similarity_threshold=0.1,
        damping=0.85,
    )
    return summarizer.summarize(texto, n_sentences=n_frases, preserve_order=True, auto_threshold=True)


def gerar_imagem_grafo(result: dict, sentences: list) -> bytes:
    """Gera a imagem do grafo e retorna como bytes PNG."""
    from graph_summarizer import SentenceGraph, pagerank as _pr

    # Reconstrói o grafo a partir dos metadados (já calculado)
    # Usa o grafo armazenado em cache via summarizer — recria para visualização
    n = result["metadados"]["grafo"]["vertices"]
    scores = result.get("scores", {})

    # Importa NetworkX só para visualização
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Usa os scores para colorir; sem arestas visíveis aqui (grafo real não re-exportado)
    fig, ax = plt.subplots(figsize=(10, 7))

    if n <= 1:
        ax.text(0.5, 0.5, "Grafo com apenas 1 vértice", ha="center", va="center")
    else:
        pos = nx.spring_layout(G, seed=42)
        node_colors = [scores.get(i, 0.0) for i in range(n)]
        cmap = plt.get_cmap("Blues")
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap,
                               node_size=300, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.4, width=0.5, ax=ax)

        # Labels apenas das frases selecionadas
        selecionadas = {item["indice"] for item in result["frases_selecionadas"]}
        labels = {
            i: f"#{i}" if i not in selecionadas else f"★{i}"
            for i in range(n)
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    ax.set_title(
        f"Grafo de Frases — {n} vértices | "
        f"{result['metadados']['grafo']['arestas']} arestas",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# LAYOUT — CABEÇALHO
# ─────────────────────────────────────────────

st.title("📚 WikiSummarizer")
st.markdown(
    "Sumarização extrativa de artigos da Wikipedia usando **grafos** e **PageRank**."
)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURAÇÕES
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Configurações")

    nivel_escolhido = st.radio(
        "Nível de resumo",
        list(NIVEIS.keys()),
        format_func=lambda k: f"{NIVEIS[k]['icon']} {k}",
        index=1,
    )
    st.caption(NIVEIS[nivel_escolhido]["descricao"])

    st.divider()

    lang = st.selectbox("Idioma da busca", ["pt", "en"], index=0)

    metodo = st.radio(
        "Método de similaridade",
        ["embeddings", "jaccard", "tfidf"],
        index=0,
        format_func=lambda m: {
            "embeddings": "Embeddings semânticos (recomendado)",
            "jaccard": "Jaccard (palavras comuns)",
            "tfidf": "TF-IDF + Cosseno",
        }[m],
    )
    if metodo == "embeddings":
        st.caption(
            "Usa o modelo `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB). "
            "Baixado automaticamente na primeira execução."
        )

    st.divider()

    mostrar_grafo = st.checkbox("Exibir visualização do grafo", value=False)
    mostrar_frases = st.checkbox("Exibir frases com pontuação", value=True)
    mostrar_keywords = st.checkbox("Exibir palavras-chave", value=True)

    st.divider()
    st.caption("Projeto EDA2 — Sumarização com Grafos")

# ─────────────────────────────────────────────
# ÁREA PRINCIPAL — INPUT
# ─────────────────────────────────────────────

col_input, col_exemplo = st.columns([3, 1])

with col_input:
    entrada = st.text_input(
        "URL ou título do artigo",
        placeholder="https://pt.wikipedia.org/wiki/Inteligência_artificial  ou  Grafos",
    )

with col_exemplo:
    st.markdown("<br>", unsafe_allow_html=True)
    exemplo = st.selectbox("Carregar exemplo", ["—"] + list(ARTIGOS_EXEMPLO.keys()), label_visibility="collapsed")
    if exemplo != "—":
        entrada = ARTIGOS_EXEMPLO[exemplo]

st.markdown("")
rodar = st.button("Gerar resumo", type="primary", use_container_width=True)  # noqa: deprecated in future

# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────

if rodar and entrada.strip():
    entrada = entrada.strip()

    # Detecta se é URL ou título direto
    if entrada.startswith("http"):
        titulo = extrair_titulo_da_url(entrada)
        lang_detectado = detectar_lang_da_url(entrada)
        lang_uso = lang_detectado
    else:
        titulo = entrada
        lang_uso = lang

    if not titulo:
        st.error("Não foi possível extrair o título da URL. Tente digitar o título diretamente.")
        st.stop()

    with st.spinner(f'Buscando artigo "{titulo}" na Wikipedia...'):
        texto = buscar_artigo(titulo, lang_uso)

    if not texto:
        st.error(f'Artigo "{titulo}" não encontrado. Verifique o título ou tente outro idioma.')
        st.stop()

    n_frases = NIVEIS[nivel_escolhido]["frases"]

    spinner_msg = (
        "Gerando embeddings e construindo grafo..."
        if metodo == "embeddings"
        else "Construindo grafo e executando PageRank..."
    )
    with st.spinner(spinner_msg):
        result = sumarizar(texto, n_frases, lang_uso, metodo)

    if "erro" in result:
        st.error(result["erro"])
        st.stop()

    meta = result["metadados"]
    perf = meta.get("performance", {})

    # ─────────────────────────────────────────────
    # RESULTADO — MÉTRICAS
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader(f"{NIVEIS[nivel_escolhido]['icon']} Resumo — {nivel_escolhido}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frases no artigo", meta["total_frases"])
    m2.metric("Frases selecionadas", meta["frases_no_resumo"])
    m3.metric("Vértices no grafo", meta["grafo"]["vertices"])
    m4.metric("Arestas no grafo", meta["grafo"]["arestas"])

    # ─────────────────────────────────────────────
    # RESULTADO — RESUMO
    # ─────────────────────────────────────────────
    st.markdown("### Resumo gerado")
    st.info(result["resumo"])

    # ─────────────────────────────────────────────
    # RESULTADO — FRASES RANQUEADAS
    # ─────────────────────────────────────────────
    if mostrar_frases:
        st.markdown("### Frases selecionadas pelo PageRank")
        for item in result["frases_selecionadas"]:
            score = item["score_pagerank"]
            idx = item["indice"]
            frase = item["frase"]
            with st.expander(f"Frase #{idx} — Score: `{score:.6f}`"):
                st.write(frase)

    # ─────────────────────────────────────────────
    # RESULTADO — PALAVRAS-CHAVE (hash table)
    # ─────────────────────────────────────────────
    if mostrar_keywords and meta.get("top_tokens"):
        st.markdown("### Palavras-chave mais frequentes")
        st.caption("Extraídas pela tabela hash de frequência de tokens.")
        tokens = meta["top_tokens"][:15]
        cols = st.columns(5)
        for i, (token, freq) in enumerate(tokens):
            cols[i % 5].metric(token, freq)

    # ─────────────────────────────────────────────
    # RESULTADO — GRAFO
    # ─────────────────────────────────────────────
    if mostrar_grafo:
        st.markdown("### Visualização do grafo")
        st.caption("Nós marcados com ★ foram selecionados para o resumo.")
        sentences_all = []
        try:
            from preprocessor import TextPreprocessor
            prep = TextPreprocessor(lang_uso)
            sentences_all = prep.segment_sentences(texto)
        except Exception:
            sentences_all = []

        img_bytes = gerar_imagem_grafo(result, sentences_all)
        st.image(img_bytes, width="stretch")

    # ─────────────────────────────────────────────
    # RESULTADO — DETALHES TÉCNICOS
    # ─────────────────────────────────────────────
    with st.expander("Detalhes técnicos do processamento"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Grafo**")
            st.write(f"- Método de similaridade: `{meta['metodo_similaridade'].upper()}`")
            st.write(f"- Threshold usado: `{meta['threshold_usado']:.4f}`")
            st.write(f"- Threshold automático: `{meta['threshold_automatico']}`")
            st.write(f"- Grau médio: `{meta['grafo']['grau_medio']:.2f}`")
            st.write(f"- Grau máximo: `{meta['grafo']['grau_max']}`")
        with col_b:
            st.markdown("**Performance**")
            st.write(f"- Tempo total: `{perf.get('tempo_segundos', '?')} s`")
            st.write(f"- Similaridades calculadas: `{perf.get('similaridades_calculadas', '?')}`")
            st.write(f"- Tokens únicos: `{perf.get('tokens_unicos', '?')}`")
            st.write(f"- Densidade do grafo: `{perf.get('densidade_grafo', '?')}`")

elif rodar and not entrada.strip():
    st.warning("Digite uma URL ou título de artigo.")
