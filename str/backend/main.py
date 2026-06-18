"""
main.py
-------
Ponto de entrada CLI do WikiSummarizer.

Uso:
    python main.py                          # usa artigo de exemplo (offline)
    python main.py --titulo "Grafos"        # busca no Wikipedia PT
    python main.py --titulo "Graphs" --lang en --metodo tfidf --frases 7
    python main.py --demo                   # roda demo completa com análise
"""
import sys
import json
import argparse
import textwrap

# Ajusta path para importar módulos do backend
sys.path.insert(0, "backend")

from wiki_fetcher import get_wiki_article, SAMPLE_TEXTS
from graph_summarizer import WikiSummarizer


def print_separator(char="═", width=70):
    print(char * width)


def print_result(result: dict, verbose: bool = False):
    """Imprime o resultado de forma legível no terminal."""
    if "erro" in result:
        print(f"\n❌ Erro: {result['erro']}")
        return

    print_separator()
    print("📄  RESUMO GERADO")
    print_separator()
    wrapper = textwrap.TextWrapper(width=70, initial_indent="  ", subsequent_indent="  ")
    print(wrapper.fill(result["resumo"]))

    print_separator("─")
    print("📊  METADADOS")
    meta = result["metadados"]
    print(f"  Frases no artigo  : {meta['total_frases']}")
    print(f"  Frases no resumo  : {meta['frases_no_resumo']}")
    print(f"  Método            : {meta['metodo_similaridade'].upper()}")
    print(f"  Threshold         : {meta['threshold_usado']}")
    g = meta["grafo"]
    print(f"  Grafo             : {g['vertices']} vértices | {g['arestas']} arestas")
    print(f"  Grau médio        : {g['grau_medio']:.2f} | Grau máx: {g['grau_max']}")

    print(f"\n  🔑 Top 10 tokens mais frequentes:")
    tokens = meta["top_tokens"][:10]
    row = "  " + "  ".join(f"{t}({c})" for t, c in tokens)
    print(textwrap.fill(row, width=70, subsequent_indent="  "))

    if verbose:
        print_separator("─")
        print("🔢  FRASES RANQUEADAS (por score PageRank)")
        for item in sorted(
            result["frases_selecionadas"],
            key=lambda x: x["score_pagerank"],
            reverse=True
        ):
            print(f"\n  [#{item['indice']:02d}] Score: {item['score_pagerank']:.6f}")
            print(textwrap.fill(
                item["frase"],
                width=68,
                initial_indent="  ",
                subsequent_indent="  "
            ))

    print_separator()


def run_demo():
    """Demonstração completa: compara Jaccard vs TF-IDF."""
    print("\n" + "═" * 70)
    print("  DEMO COMPLETA — WikiSummarizer MVP")
    print("═" * 70)

    for topic, text in SAMPLE_TEXTS.items():
        for method in ["jaccard", "tfidf"]:
            print(f"\n{'─'*70}")
            print(f"  Tópico : {topic.replace('_', ' ').title()}")
            print(f"  Método : {method.upper()}")
            print(f"{'─'*70}")

            summarizer = WikiSummarizer(
                lang="pt",
                similarity_method=method,
                similarity_threshold=0.08,
                damping=0.85,
            )
            result = summarizer.summarize(text, n_sentences=4)
            print_result(result, verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description="WikiSummarizer — Sumarização extrativa com grafos e PageRank"
    )
    parser.add_argument("--titulo",   type=str,  default=None,       help="Título do artigo Wikipedia")
    parser.add_argument("--lang",     type=str,  default="pt",       help="Idioma: pt | en")
    parser.add_argument("--metodo",   type=str,  default="jaccard",  help="Similaridade: jaccard | tfidf")
    parser.add_argument("--frases",   type=int,  default=5,          help="Nº de frases no resumo")
    parser.add_argument("--threshold",type=float,default=0.1,        help="Threshold de similaridade")
    parser.add_argument("--demo",     action="store_true",           help="Roda demo completa")
    parser.add_argument("--verbose",  action="store_true",           help="Mostra frases ranqueadas")
    parser.add_argument("--json",     action="store_true",           help="Saída em JSON")
    parser.add_argument("--export", type=str, default=None,          help="Exportar resumo (formatos: md, json, html)")
    parser.add_argument("--viz", action="store_true",                help="Visualizar grafo de similaridade")
    parser.add_argument("--output", type=str, default="output",      help="Nome base para arquivos de saída")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    # Obtém texto
    if args.titulo:
        text = get_wiki_article(args.titulo, lang=args.lang)
        if not text:
            print(f"Artigo '{args.titulo}' não encontrado. Usando exemplo offline.")
            text = list(SAMPLE_TEXTS.values())[0]
    else:
        print("Nenhum título fornecido — usando artigo de exemplo (offline).")
        text = list(SAMPLE_TEXTS.values())[0]

    summarizer = WikiSummarizer(
        lang=args.lang,
        similarity_method=args.metodo,
        similarity_threshold=args.threshold,
        damping=0.85,
    )

    result = summarizer.summarize(text, n_sentences=args.frases)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()