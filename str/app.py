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

from backend.wiki_fetcher import get_wiki_article, SAMPLE_TEXTS
from backend.graph_summarizer import WikiSummarizer

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
    "Personalizado (%)": {
        "frases": None,
        "descricao": "Define o tamanho do resumo como porcentagem do artigo.",
        "icon": "🎚️",
    },
}

ARTIGOS_EXEMPLO = {
    "Sistema Solar": "Sistema Solar",
    "Inteligência Artificial": "Inteligência artificial",
    "Grafos (matemática)": "Teoria dos grafos",
    "Revolução Industrial": "Revolução Industrial",
    "Biomas do Brasil": "Biomas do Brasil",
}

# Opções do seletor customizado: URL + exemplos (separador é visual apenas no HTML)
OPCOES_EXEMPLOS = list(ARTIGOS_EXEMPLO.keys())
OPCOES_SELETOR = ["URL"] + OPCOES_EXEMPLOS

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
        host = urlparse(url).hostname  # pt.wikipedia.org or None
        if not host:
            return "pt"
        return host.split(".")[0]
    except Exception:
        return "pt"


@st.cache_data(show_spinner=False)
def buscar_artigo(titulo: str, lang: str) -> str | None:
    return get_wiki_article(titulo, lang=lang)


@st.cache_data(show_spinner=False)
def sumarizar(
    texto: str,
    n_frases: int | None,
    summary_percent: float | None,
    lang: str,
    metodo: str,
    threshold_manual: float,
    threshold_strategy: str,
    target_density: float,
    damping: float,
    pagerank_max_iter: int,
    pagerank_tol: float,
    selection_strategy: str,
    diversity_alpha: float,
    include_graph: bool,
) -> dict:
    summarizer = WikiSummarizer(
        lang=lang,
        similarity_method=metodo,
        similarity_threshold=threshold_manual,
        damping=damping,
        pagerank_max_iter=pagerank_max_iter,
        pagerank_tol=pagerank_tol,
    )
    return summarizer.summarize(
        texto,
        n_sentences=n_frases or 5,
        preserve_order=True,
        auto_threshold=threshold_strategy != "manual",
        summary_percent=summary_percent,
        threshold_strategy=threshold_strategy,
        target_density=target_density,
        selection_strategy=selection_strategy,
        diversity_alpha=diversity_alpha,
        include_graph=include_graph,
    )


def gerar_imagem_grafo(result: dict, sentences: list) -> bytes:
    """Gera a imagem do grafo e retorna como bytes PNG."""
    n = result["metadados"]["grafo"]["vertices"]
    scores = result.get("scores", {})
    graph_data = result.get("grafo_serializado", {})
    edges = graph_data.get("arestas", [])

    # Importa NetworkX só para visualização
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for edge in edges:
        G.add_edge(edge["origem"], edge["destino"], weight=edge["peso"])

    fig, ax = plt.subplots(figsize=(10, 7))

    if n <= 1:
        ax.text(0.5, 0.5, "Grafo com apenas 1 vértice", ha="center", va="center")
    else:
        selecionadas = {item["indice"] for item in result["frases_selecionadas"]}
        pos = nx.spring_layout(G, seed=42, k=1.2, iterations=80)
        widths = [0.5 + 4 * data["weight"] for _, _, data in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", alpha=0.35, width=widths, ax=ax)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[scores.get(i, 0.0) for i in range(n)],
            cmap=plt.get_cmap("viridis"),
            node_size=[560 if i in selecionadas else 280 for i in range(n)],
            edgecolors=["#f39c12" if i in selecionadas else "#263238" for i in range(n)],
            linewidths=[2.0 if i in selecionadas else 0.4 for i in range(n)],
            alpha=0.9,
            ax=ax,
        )
        label_nodes = set(range(n)) if n <= 20 else selecionadas
        labels = {i: f"#{i}" if i not in selecionadas else f"★{i}" for i in label_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    ax.set_title(
        f"Grafo de Frases — {n} vértices | "
        f"{result['metadados']['grafo']['arestas']} arestas | "
        f"densidade {result['metadados']['grafo']['densidade']:.3f}",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def renderizar_resumo(titulo: str, resumo: str, meta: dict, lang: str):
    """Renderiza o resumo com componentes nativos do Streamlit."""
    st.markdown(f"#### {titulo or 'Artigo'}")
    st.caption(
        f"{meta.get('frases_no_resumo', '?')} de {meta.get('total_frases', '?')} frases | "
        f"{meta.get('metodo_similaridade', '?').upper()} | {lang.upper()}"
    )
    st.info(resumo)


def detail_row(label: str, value, tooltip: str):
    st.markdown(f"**{label}:** `{value}`", help=tooltip)


# ─────────────────────────────────────────────
# SESSION STATE — inicialização
# ─────────────────────────────────────────────

if "seletor_escolha" not in st.session_state:
    st.session_state["seletor_escolha"] = "URL"

if "url_travada" not in st.session_state:
    st.session_state["url_travada"] = False

# _entrada_widget é a key direta do text_input — escrevemos nela para forçar o valor
if "_entrada_widget" not in st.session_state:
    st.session_state["_entrada_widget"] = ""


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
    summary_percent = None
    if nivel_escolhido == "Personalizado (%)":
        summary_percent = st.slider(
            "Porcentagem do artigo",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="O backend calcula ceil(total de frases × porcentagem).",
        )

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

    with st.expander("Avançado"):
        threshold_label = st.selectbox(
            "Estratégia de threshold",
            ["Densidade-alvo", "Média + desvio", "Manual"],
            index=0,
            help="Define como a similaridade mínima vira aresta: densidade-alvo, média+desvio ou valor manual.",
        )
        threshold_strategy = {"Densidade-alvo": "auto_density", "Média + desvio": "mean_std", "Manual": "manual"}[threshold_label]

        target_density_percent = st.slider(
            "Densidade-alvo do grafo (%)",
            min_value=1,
            max_value=40,
            value=12,
            step=1,
            disabled=threshold_strategy != "auto_density",
            help="Percentual aproximado de pares de frases mantidos como arestas.",
        )
        target_density = target_density_percent / 100

        threshold_manual = st.slider(
            "Threshold manual",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            disabled=threshold_strategy != "manual",
            help="Similaridade mínima para criar uma aresta no modo Manual.",
        )

        damping = st.slider(
            "Damping do PageRank",
            min_value=0.50,
            max_value=0.95,
            value=0.85,
            step=0.01,
            help="Peso dado a seguir as conexões do grafo no PageRank.",
        )
        pagerank_max_iter = st.number_input(
            "Máximo de iterações",
            min_value=20,
            max_value=500,
            value=100,
            step=10,
            help="Limite de ciclos do PageRank antes de parar.",
        )
        pagerank_tol = st.selectbox(
            "Tolerância de convergência",
            [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            index=2,
            format_func=lambda v: f"{v:.0e}",
            help="Diferença mínima entre iterações para considerar convergência.",
        )

        selection_label = st.selectbox(
            "Estratégia de seleção",
            ["MMR (diversidade)", "PageRank + filtro"],
            index=0,
            help="MMR favorece diversidade; PageRank + filtro usa o baseline com remoção de redundância.",
        )
        selection_strategy = {"MMR (diversidade)": "mmr", "PageRank + filtro": "redundancy"}[selection_label]
        diversity_alpha = st.slider(
            "Alpha do MMR",
            min_value=0.50,
            max_value=0.95,
            value=0.85,
            step=0.01,
            disabled=selection_strategy != "mmr",
            help="Peso do PageRank no MMR; menor alpha aumenta diversidade.",
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

# Separador visual — item que, se selecionado, é ignorado e voltamos para "URL"
SEPARADOR = "――――――――――――――――――――――"
OPCOES_SELETOR_FULL = ["URL", SEPARADOR] + OPCOES_EXEMPLOS

# CSS que desabilita visualmente o item do separador no <select> nativo do browser.
# O Streamlit renderiza selectbox como um <select> HTML nativo, então nth-child funciona.
# Índice 2 = segundo <option> = o separador (1-based no CSS).
st.markdown("""
<style>
/* Torna o separador visualmente não-clicável dentro do selectbox nativo */
[data-testid="stSelectbox"] select option:nth-child(2) {
    color: #999;
    font-size: 11px;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

col_input, col_exemplo = st.columns([3, 1])


def _ao_mudar_seletor():
    escolha = st.session_state["_seletor_widget"]

    # Se o usuário conseguiu selecionar o separador, volta para "URL"
    if escolha == SEPARADOR:
        st.session_state["_seletor_widget"] = "URL"
        st.session_state["seletor_escolha"] = "URL"
        st.session_state["url_travada"] = False
        st.session_state["_entrada_widget"] = ""
        return

    st.session_state["seletor_escolha"] = escolha

    if escolha == "URL":
        st.session_state["url_travada"] = False
        st.session_state["_entrada_widget"] = ""
    else:
        # Artigo de exemplo: escreve direto na key do text_input para forçar o valor
        st.session_state["_entrada_widget"] = ARTIGOS_EXEMPLO[escolha]
        st.session_state["url_travada"] = True


with col_exemplo:
    st.markdown("<br>", unsafe_allow_html=True)

    # Índice atual no seletor — garante que o selectbox reflita a escolha persistida
    idx_atual = OPCOES_SELETOR_FULL.index(st.session_state["seletor_escolha"])

    st.selectbox(
        "Carregar exemplo",
        options=OPCOES_SELETOR_FULL,
        index=idx_atual,
        key="_seletor_widget",
        on_change=_ao_mudar_seletor,
        label_visibility="collapsed",
    )

with col_input:
    # text_input usa a mesma key que escrevemos em _ao_mudar_seletor,
    # então o Streamlit exibe o valor correto sem precisar de rerun extra.
    st.text_input(
        "URL ou título do artigo",
        placeholder="https://pt.wikipedia.org/wiki/Inteligência_artificial  ou  Grafos",
        disabled=st.session_state["url_travada"],
        key="_entrada_widget",
    )

    if st.session_state["url_travada"]:
        st.caption("🔒 Artigo de exemplo selecionado. Escolha **URL** no seletor para digitar livremente.")

# Valor final: lê diretamente da key do widget
entrada_final = st.session_state.get("_entrada_widget", "")

st.markdown("")
rodar = st.button("Gerar resumo", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────

if rodar and entrada_final.strip():
    entrada_proc = entrada_final.strip()

    # Detecta se é URL ou título direto
    if entrada_proc.startswith("http"):
        titulo = extrair_titulo_da_url(entrada_proc)
        lang_detectado = detectar_lang_da_url(entrada_proc)
        lang_uso = lang_detectado
    else:
        titulo = entrada_proc
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
        result = sumarizar(
            texto,
            n_frases,
            summary_percent,
            lang_uso,
            metodo,
            threshold_manual,
            threshold_strategy,
            target_density,
            damping,
            int(pagerank_max_iter),
            pagerank_tol,
            selection_strategy,
            diversity_alpha,
            mostrar_grafo,
        )

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
    renderizar_resumo(titulo, result["resumo"], meta, lang_uso)

    # ─────────────────────────────────────────────
    # RESULTADO — FRASES RANQUEADAS
    # ─────────────────────────────────────────────
    if mostrar_frases:
        st.markdown("### Frases selecionadas")
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
        st.caption("Arestas usam os pesos reais de similaridade; nós marcados com ★ foram selecionados.")
        sentences_all = []
        try:
            from backend.preprocessor import TextPreprocessor
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
        threshold_meta = meta.get("threshold", {})
        pagerank_meta = meta.get("pagerank", {})
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Grafo**")
            for label, value, tip in [
                ("Método de similaridade", meta["metodo_similaridade"].upper(), "Forma usada para calcular o peso entre duas frases."),
                ("Estratégia de threshold", threshold_meta.get("strategy", "?"), "Regra usada para decidir o corte mínimo de similaridade."),
                ("Threshold usado", f"{meta['threshold_usado']:.4f}", "Similaridade mínima exigida para conectar duas frases."),
                ("Média das similaridades", threshold_meta.get("mean_similarity", "?"), "Média entre todos os pares de frases avaliados."),
                ("Desvio das similaridades", threshold_meta.get("std_similarity", "?"), "Variação das similaridades em torno da média."),
                ("Densidade do grafo", meta["grafo"].get("densidade", "?"), "Proporção de arestas existentes em relação ao máximo possível."),
                ("Grau médio", f"{meta['grafo']['grau_medio']:.2f}", "Número médio de conexões por frase."),
                ("Grau máximo", meta["grafo"]["grau_max"], "Maior número de conexões em uma frase."),
                ("Nós isolados", meta["grafo"].get("nos_isolados", "?"), "Frases sem conexão com outras frases."),
                ("Componentes conectados", meta["grafo"].get("componentes_conectados", "?"), "Quantidade de grupos separados no grafo."),
            ]:
                detail_row(label, value, tip)
        with col_b:
            st.markdown("**PageRank e seleção**")
            for label, value, tip in [
                ("Damping", pagerank_meta.get("damping", "?"), "Peso dado às arestas no PageRank."),
                ("Iterações", pagerank_meta.get("iteracoes", "?"), "Ciclos executados até parar."),
                ("Convergiu", pagerank_meta.get("convergiu", "?"), "Indica se atingiu a tolerância configurada."),
                ("Delta final", pagerank_meta.get("delta_final", "?"), "Variação dos scores na última iteração."),
                ("Soma dos scores", pagerank_meta.get("soma_scores", "?"), "Soma final dos scores PageRank."),
                ("Estratégia de seleção", meta.get("selection_strategy", "?"), "Método usado para escolher as frases finais."),
                ("Alpha MMR", meta.get("diversity_alpha", "?"), "Peso do PageRank na seleção MMR."),
                ("Frases redundantes removidas", meta.get("frases_redundantes_removidas", "?"), "Candidatas descartadas por redundância."),
            ]:
                detail_row(label, value, tip)

        st.markdown("**Performance**")
        col_c, col_d = st.columns(2)
        with col_c:
            detail_row("Tempo total", f"{perf.get('tempo_segundos', '?')} s", "Tempo total de processamento.")
            detail_row("Similaridades calculadas", perf.get("similaridades_calculadas", "?"), "Pares de frases comparados.")
        with col_d:
            detail_row("Tokens únicos", perf.get("tokens_unicos", "?"), "Termos distintos após pré-processamento.")
            detail_row("Média de tokens por frase", perf.get("media_tokens_por_frase", "?"), "Média de termos úteis por frase.")

elif rodar and not entrada_final.strip():
    st.warning("Digite uma URL ou título de artigo.")
