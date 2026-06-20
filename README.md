# 📚 WikiSummarizer

Sumarização extrativa de artigos da Wikipedia baseada em **grafos** e **PageRank**, desenvolvida para a disciplina **EDA2 — Estrutura de Dados e Algoritmos 2** (UnB/FGA).

> Disciplina: PLN aplicado com modelagem em grafos
> Área de aplicação: Educação (síntese de conteúdo enciclopédico)

---

## 1. Descrição do problema

Ler artigos longos da Wikipedia consome tempo. O objetivo deste projeto é **gerar automaticamente um resumo extrativo**, composto por frases reais do texto original, não reescritas, que preserve as ideias centrais do artigo.

O sistema recebe um artigo (via título ou URL da Wikipedia), processa o texto e devolve um conjunto reduzido de frases consideradas as mais relevantes, mantendo a ordem em que aparecem no texto original para preservar a coerência de leitura.

**Entrada:** artigos da Wikipedia em português ou inglês (título ou URL).
**Saída:** resumo extrativo + métricas do grafo gerado (vértices, arestas, densidade, grau médio) + palavras-chave mais frequentes.

---

## 2. Modelagem do grafo

| Elemento | Definição |
|---|---|
| **Vértices** | Cada frase do artigo é um vértice. |
| **Arestas** | Criada entre duas frases quando a similaridade textual entre elas é maior ou igual a um *threshold* (fixo ou adaptativo). |
| **Peso** | Valor da similaridade entre as duas frases conectadas. |
| **Tipo de grafo** | Não direcionado e ponderado (a similaridade é simétrica: `sim(A,B) = sim(B,A)`). |
| **Representação interna** | Lista de adjacência (`dict` de `dict`), `adj[i][j] = peso`. |

Uma frase é considerada relevante quando está fortemente conectada a outras frases também relevantes — exatamente a lógica do PageRank aplicada a texto (também conhecida como *TextRank*).

### 2.1 Métodos de similaridade suportados

O projeto implementa **três** formas de calcular o peso das arestas, selecionáveis na interface:

1. **Embeddings semânticos** (padrão/recomendado) — similaridade do cosseno entre vetores gerados pelo modelo `paraphrase-multilingual-MiniLM-L12-v2` (`sentence-transformers`).
2. **TF-IDF + cosseno** — implementado do zero (sem `sklearn`): `TF(t,d) = freq(t,d)/max_freq(d)`, `IDF(t) = log(N/(df(t)+1)) + 1`.
3. **Jaccard** — `|A ∩ B| / |A ∪ B|` sobre os conjuntos de tokens lematizados de cada frase.

### 2.2 Threshold de criação de arestas

Para evitar um grafo denso demais (com relações fracas), o threshold pode ser:

- **Fixo**, definido manualmente (ex.: `0.10`);
- **Adaptativo**, calculado a partir da distribuição estatística das similaridades do próprio artigo: `threshold = média + k × desvio_padrão`, limitado a um intervalo seguro (`[0.05, 0.5]` para Jaccard; `[0.2, 0.85]` para embeddings).

> **Por que adaptativo?** Artigos diferentes têm densidades de similaridade muito distintas — um threshold fixo pode deixar um artigo com grafo quase vazio e outro com grafo quase completo. O threshold adaptativo ajusta-se à distribuição real de cada texto.

---

## 3. Estruturas de dados utilizadas

Além do **grafo** (estrutura central, lista de adjacência), o projeto utiliza:

| Estrutura | Onde é usada | Por quê |
|---|---|---|
| **Tabela hash (`dict`)** | `freq_table`, `global_freq`, `df` em `compute_tfidf()` | Armazenar frequência de tokens por frase e frequência global, com acesso O(1) — essencial para o cálculo eficiente de TF-IDF. |
| **Fila de prioridade (`heapq`)** | `top_k_sentences()` | Selecionar as *k* frases com maior score de PageRank sem precisar ordenar todos os *n* scores — complexidade O(n log k) em vez de O(n log n). |

---

## 4. Algoritmos principais (implementados pelo grupo, sem bibliotecas prontas)

### 4.1 PageRank ponderado

Implementado integralmente em `pagerank()`, sem uso de `networkx.pagerank` ou qualquer biblioteca de grafos para o ranqueamento.

```
PR(v) = (1 - d)/N + d × Σ [ PR(u) × w(u,v) / Σ_k w(u,k) ]
```

- **Fator de amortecimento (d):** 0.85
- **Critério de parada:** máximo de 100 iterações **ou** variação L1 entre iterações menor que `1e-6`
- **Inicialização:** distribuição uniforme (`1/N` para todos os vértices)

### 4.2 TF-IDF

Implementado do zero em `compute_tfidf()` (sem `sklearn`), conforme fórmulas descritas na seção 2.1.

### 4.3 Similaridade de Jaccard e Cosseno

Implementadas em `jaccard_similarity()` e `cosine_tfidf_similarity()`, sem dependências externas para o cálculo.

> **Nota sobre o NetworkX:** a biblioteca `networkx` é utilizada **exclusivamente para visualização** do grafo na interface (método `to_networkx()` e renderização com `matplotlib`), nunca para os algoritmos de construção do grafo ou de ranqueamento. Os pesos, arestas e o PageRank exibidos na interface são sempre os calculados pela implementação própria do grupo.

---

## 5. Pipeline completo

```
1. Buscar artigo na Wikipedia (título ou URL)
2. Segmentar o texto em frases (spaCy)
3. Tokenizar, remover stopwords e lematizar cada frase (spaCy)
4. Calcular similaridade entre todos os pares de frases
   (embeddings, TF-IDF ou Jaccard)
5. Construir o grafo:
   - vértices = frases
   - arestas = similaridade ≥ threshold (fixo ou adaptativo)
   - pesos = valor da similaridade
6. Executar PageRank ponderado (implementação própria)
7. Selecionar as N frases de maior score via fila de prioridade (heapq)
8. Reordenar as frases selecionadas pela posição original no texto
9. Concatenar as frases → resumo final
```

---

## 6. Exemplos de entrada e saída

**Entrada (título):**
```
Inteligência artificial
```

**Entrada (URL):**
```
https://pt.wikipedia.org/wiki/Inteligência_artificial
```

**Saída (resumida):**
- Resumo extrativo com N frases (3, 5 ou 9, conforme o nível escolhido)
- Métricas do grafo: total de frases, vértices, arestas, grau médio, grau máximo, densidade
- Lista de frases selecionadas com seu score de PageRank
- Top palavras-chave por frequência (via tabela hash)
- Visualização opcional do grafo, com as frases selecionadas marcadas (★)

---

## 7. Como executar

### 7.1 Instalação

```bash
pip install -r requirements.txt
```

### 7.2 Execução

```bash
cd str
streamlit run app.py
```

A interface abre no navegador. Escolha um artigo (URL ou título), o nível de resumo, o método de similaridade e clique em **"Gerar resumo"**.

---

## 8. Estrutura do projeto

```
EDA2-Sumarizacao
├─ README.md
└─ str
   ├─ app.py                    # Interface Streamlit
   ├─ backend
   │  ├─ embedder.py            # Geração de embeddings semânticos
   │  ├─ exporters.py
   │  ├─ graph_summarizer.py    # Grafo, PageRank, TF-IDF, fila de prioridade
   │  ├─ main.py
   │  ├─ preprocessor.py        # Pré-processamento de texto (spaCy)
   │  ├─ visualization.py
   │  └─ wiki_fetcher.py        # Busca de artigos na Wikipedia
   ├─ requirements.txt
   └─ test
      └─ test_wikisummarizer.py
```

---

## 9. Análise dos resultados

> *(Preencher após rodar os testes finais com múltiplos artigos.)*

Pontos a abordar nesta seção / no slide de análise:

- Comparação dos 3 métodos de similaridade (embeddings vs. TF-IDF vs. Jaccard) em termos de coerência do resumo gerado para um mesmo artigo.
- Efeito do threshold adaptativo na densidade do grafo entre artigos de tamanhos diferentes.
- Relação entre grau médio do grafo e qualidade percebida do resumo (grafos muito esparsos tendem a gerar resumos menos representativos; grafos muito densos tendem a perder seletividade do PageRank).
- Tempo de execução por método (embeddings é mais lento por gerar vetores semânticos; Jaccard é o mais rápido).

---

## 10. Uso de LLM no desenvolvimento

LLMs (incluindo Claude, da Anthropic) foram utilizados como apoio durante o desenvolvimento para:

- Discussão e validação da modelagem do grafo (definição de vértices, arestas, pesos e threshold) junto ao professor da disciplina;
- Revisão e depuração de erros de ambiente/dependências (ex.: conflitos de `torch`/`torchvision`);
- Estruturação da documentação técnica (este README).

A **implementação dos algoritmos principais de grafos** (construção do grafo, similaridade, TF-IDF e PageRank) foi feita integralmente pelo grupo, sem geração de código por LLM para essas partes, em conformidade com o item 7 das regras do trabalho.

---

## 11. Equipe

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/JoaoEduardoP">
        <img src="https://avatars.githubusercontent.com/JoaoEduardoP" width="100" height="100" style="border-radius:50%"/><br/>
        <sub><b>Caio Rocha</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/audittmega">
        <img src="https://avatars.githubusercontent.com/audittmega" width="100" height="100" style="border-radius:50%"/><br/>
        <sub><b>Eduardo Morais</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/TiagoBalieiro">
        <img src="https://avatars.githubusercontent.com/TiagoBalieiro" width="100" height="100" style="border-radius:50%"/><br/>
        <sub><b>Fábio Torres</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/JoaoEduardoP">
        <img src="https://avatars.githubusercontent.com/JoaoEduardoP" width="100" height="100" style="border-radius:50%"/><br/>
        <sub><b>João Eduardo</b></sub>
      </a>
    </td>
  </tr>
</table>


---

## 12. Licença / Disciplina

Projeto acadêmico desenvolvido para a disciplina **EDA2** — Universidade de Brasília (UnB), Faculdade do Gama (FGA).