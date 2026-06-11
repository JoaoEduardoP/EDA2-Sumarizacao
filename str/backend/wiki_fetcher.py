"""
wiki_fetcher.py
---------------
Busca e armazena artigos da Wikipedia em cache local (hash table em memória).
Suporte a PT e EN com fallback automático.
"""

import time
from typing import Optional
import wikipediaapi

# Cache em memória: dict (hash table) título → texto
_cache: dict = {}


def get_wiki_article(title: str, lang: str = "pt") -> Optional[str]:
    """
    Busca artigo da Wikipedia por título.
    Usa cache local para evitar requisições repetidas.
    
    Args:
        title: Título do artigo (ex: "Inteligência artificial")
        lang:  Código de idioma ("pt" ou "en")
    
    Returns:
        Texto completo do artigo ou None se não encontrado.
    """
    cache_key = f"{lang}:{title.lower()}"

    if cache_key in _cache:
        print(f"  [cache] '{title}' recuperado da hash table local")
        return _cache[cache_key]

    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="WikiSummarizer-MVP/1.0 (estudo de estrutura de dados)"
    )

    print(f"  [wikipedia] Buscando '{title}' em [{lang.upper()}]...")
    page = wiki.page(title)

    if not page.exists():
        print(f"  [erro] Artigo '{title}' não encontrado.")
        return None

    text = page.text
    # Salva no cache (hash table)
    _cache[cache_key] = text
    print(f"  [ok] {len(text):,} caracteres | {len(text.split())} palavras")
    return text


def list_cached() -> list:
    """Retorna títulos atualmente em cache."""
    return list(_cache.keys())


# Artigos de teste pré-definidos (para demonstração offline)
SAMPLE_TEXTS = {
    "sistema_solar": """
        O Sistema Solar é o conjunto formado pelo Sol e todos os corpos celestes que orbitam ao seu redor.
        O Sol contém 99,86% de toda a massa do Sistema Solar e é a sua principal fonte de energia.
        Os planetas do Sistema Solar são divididos em planetas rochosos e planetas gasosos.
        Mercúrio, Vênus, Terra e Marte são os planetas rochosos do Sistema Solar.
        Júpiter, Saturno, Urano e Netuno são os planetas gasosos do Sistema Solar.
        A Terra é o único planeta conhecido que abriga vida no Sistema Solar.
        Júpiter é o maior planeta do Sistema Solar, com mais de 1.300 vezes o volume da Terra.
        Saturno é famoso pelos seus anéis compostos principalmente por gelo e partículas de rocha.
        O cinturão de asteroides localiza-se entre Marte e Júpiter e contém milhões de corpos rochosos.
        A Lua é o único satélite natural da Terra e o quinto maior do Sistema Solar.
    """,
    
    "revolucao_industrial": """
        A Revolução Industrial foi o período de grande desenvolvimento tecnológico que teve início na Inglaterra no século XVIII.
        A máquina a vapor, aperfeiçoada por James Watt, foi a principal inovação da Primeira Revolução Industrial.
        A indústria têxtil foi uma das primeiras a ser transformada pelas novas máquinas de tear mecânico.
        A Revolução Industrial provocou um êxodo rural em massa, com trabalhadores se deslocando para as cidades.
        As condições de trabalho nas fábricas eram precárias, com jornadas de até 16 horas diárias.
        A Segunda Revolução Industrial, no final do século XIX, foi marcada pela eletricidade e pelo motor a combustão.
        Henry Ford introduziu a linha de montagem, revolucionando a produção industrial em massa.
        A Revolução Industrial transformou a sociedade feudal em capitalista, com novas classes sociais.
        O luddismo foi um movimento que destruía máquinas por considerá-las causadoras de desemprego.
        As consequências ambientais da Revolução Industrial incluem a poluição e as mudanças climáticas.
    """,
    
    "biomas_brasil": """
        O Brasil possui seis biomas principais: Amazônia, Cerrado, Mata Atlântica, Caatinga, Pampa e Pantanal.
        A Amazônia é o maior bioma brasileiro e a maior floresta tropical do mundo.
        O Cerrado é considerado o berço das águas do Brasil por abrigar nascentes de importantes rios.
        A Mata Atlântica é um dos biomas mais ameaçados do mundo, com apenas 12% de sua cobertura original.
        A Caatinga é o único bioma exclusivamente brasileiro, presente apenas no Nordeste do país.
        O Pantanal é a maior planície alagável do mundo, com rica biodiversidade de aves e peixes.
        O Pampa é o bioma mais ao sul do Brasil, com paisagens de campos e coxilhas.
        A fauna brasileira inclui espécies ameaçadas como a arara-azul, o mico-leão-dourado e a onça-pintada.
        O desmatamento na Amazônia tem causas como a pecuária extensiva e a mineração ilegal.
        A conservação dos biomas brasileiros é essencial para a regulação climática do planeta.
    """,
}