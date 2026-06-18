# exporters.py
import json
from typing import Dict
from pathlib import Path

def export_to_markdown(result: Dict, output_file: str = "resumo.md"):
    """Exporta resumo para formato Markdown."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Resumo Gerado\n\n")
        f.write(f"{result['resumo']}\n\n")
        f.write(f"## Metadados\n\n")
        f.write(f"| Campo | Valor |\n")
        f.write(f"|-------|-------|\n")
        f.write(f"| Total de frases | {result['metadados']['total_frases']} |\n")
        f.write(f"| Frases no resumo | {result['metadados']['frases_no_resumo']} |\n")
        f.write(f"| Método de similaridade | {result['metadados']['metodo_similaridade']} |\n")
        f.write(f"| Threshold | {result['metadados']['threshold_usado']} |\n")
        f.write(f"| Vértices no grafo | {result['metadados']['grafo']['vertices']} |\n")
        f.write(f"| Arestas no grafo | {result['metadados']['grafo']['arestas']} |\n\n")
        
        f.write(f"## Frases Selecionadas\n\n")
        for item in result['frases_selecionadas']:
            f.write(f"- **Score {item['score_pagerank']:.6f}**: {item['frase']}\n")

def export_to_json(result: Dict, output_file: str = "resumo.json"):
    """Exporta resultado completo para JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def export_to_html(result: Dict, output_file: str = "resumo.html"):
    """Exporta resumo para HTML com formatação."""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WikiSummarizer - Resumo</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        .summary {{ background: #f0f7ff; padding: 20px; border-radius: 10px; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        .sentences {{ margin-top: 20px; }}
        .sentence {{ padding: 8px; margin: 5px 0; border-left: 3px solid #4CAF50; }}
        .score {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>📄 WikiSummarizer - Resumo</h1>
    <div class="summary">
        <h2>Resumo Gerado</h2>
        <p>{summary}</p>
    </div>
    <div class="metadata">
        <h2>Metadados</h2>
        <ul>
            <li>Total de frases: {total_frases}</li>
            <li>Frases no resumo: {frases_resumo}</li>
            <li>Método: {metodo}</li>
            <li>Threshold: {threshold}</li>
        </ul>
    </div>
    <div class="sentences">
        <h2>Frases Selecionadas</h2>
        {sentences_html}
    </div>
</body>
</html>
    """
    sentences_html = ""
    for item in result['frases_selecionadas']:
        sentences_html += f"""
        <div class="sentence">
            <span class="score">Score: {item['score_pagerank']:.6f} (índice {item['indice']})</span>
            <p>{item['frase']}</p>
        </div>
        """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template.format(
            summary=result['resumo'].replace('\n', '<br>'),
            total_frases=result['metadados']['total_frases'],
            frases_resumo=result['metadados']['frases_no_resumo'],
            metodo=result['metadados']['metodo_similaridade'],
            threshold=result['metadados']['threshold_usado'],
            sentences_html=sentences_html
        ))