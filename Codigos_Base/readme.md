# Aceleração Massivamente Paralela de um Agente de IA para Truco

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Numba](https://img.shields.io/badge/Numba%20(CUDA)-0.59-green.svg)
![Joblib](https://img.shields.io/badge/Joblib-1.4-orange.svg)
![NumPy](https://img.shields.io/badge/Numpy-1.26-blueviolet.svg)

## Resumo

[cite_start]Este repositório detalha o desenvolvimento e a avaliação de desempenho de um agente de inteligência artificial (IA) para o jogo de cartas Truco.  [cite_start]O núcleo do agente de IA baseia-se em Simulações de Monte Carlo (SMC), evoluindo para uma implementação completa de Monte Carlo Tree Search (MCTS), para lidar com a natureza de informação imperfeita do jogo.  [cite_start]O foco principal do projeto é a exploração e comparação de diferentes paradigmas de computação paralela para acelerar a tomada de decisão do agente. 

Foram desenvolvidas e comparadas três implementações distintas:
1.  [cite_start]**CPU Single-Core:** Uma versão sequencial para estabelecer a linha de base de desempenho. 
2.  [cite_start]**CPU Multi-Core:** Uma versão paralela de memória compartilhada utilizando múltiplos núcleos de CPU através da biblioteca `joblib`. 
3.  [cite_start]**GPU (Massivamente Paralela):** Uma versão com aceleração por GPU implementada com `Numba`, explorando o paralelismo massivo dos processadores gráficos para a etapa de *rollout* do MCTS. 

[cite_start]O desempenho foi rigorosamente avaliado em termos de tempo de execução e speedup, culminando em um torneio entre as IAs para testar sua força estratégica sob um limite de tempo. 

## Visão Geral do Projeto

### O Desafio: O Jogo de Truco

[cite_start]Foi escolhido o Truco, um dos jogos de cartas mais populares do Brasil, como campo de provas computacional.  [cite_start]O jogo é caracterizado não apenas pela sorte e habilidade, mas também por blefes e, crucialmente para este projeto, pela **informação imperfeita**.  [cite_start]Um jogador conhece apenas suas próprias cartas e deve inferir as mãos dos oponentes com base em suas ações.  [cite_start]Este projeto foca na variante Truco Paulista. 

### A Abordagem de IA: Monte Carlo Tree Search (MCTS)

[cite_start]A abordagem de IA escolhida é a **Monte Carlo Tree Search (MCTS)**.  MCTS é um algoritmo de busca heurística que se tornou o estado da arte para jogos de tabuleiro e de cartas. [cite_start]Ele constrói uma árvore de busca assimétrica, focando nos movimentos mais promissores, e usa *rollouts* (simulações aleatórias rápidas) para estimar o valor de cada movimento.  [cite_start]Para lidar com a informação imperfeita do Truco, a técnica de **determinization** é empregada, onde, para cada simulação, as cartas desconhecidas são distribuídas de forma aleatória e consistente com o estado atual do jogo. 

## Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Computação Paralela em CPU:** `Joblib`
* **Aceleração em GPU:** `Numba (CUDA)`
* **Computação Numérica:** `NumPy`
* **Análise de Dados e Plotagem:** `Pandas`, `Matplotlib`, `Seaborn`

## Estrutura do Repositório

```
.
├── agente_gpu.py           # Agente MCTS com aceleração em GPU (Numba)
├── agente_mcts_multi.py    # Agente MCTS para CPU (single e multi-core com Joblib)
├── benchmark_runner.py     # Script para rodar o benchmark em larga escala
├── gpu_utils.py            # Funções auxiliares para o agente GPU (achatamento de dados)
├── logica.py               # Contém as regras e a lógica central do jogo de Truco
├── tournament.py           # Script para executar o torneio final entre as IAs
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## Como Executar

### Pré-requisitos
1.  Python 3.10 ou superior.
2.  Git.
3.  Uma placa de vídeo NVIDIA com suporte a CUDA.
4.  **NVIDIA CUDA Toolkit instalado.** É crucial para a execução do agente GPU. Faça o download em [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). A instalação "Express" é recomendada.

### Instalação
1.  Clone o repositório:
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DA_PASTA>
    ```
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    # No Windows
    .\.venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Executando os Testes
O projeto possui dois modos principais de execução:

1.  **Benchmark em Larga Escala:** Para comparar a performance bruta de cada arquitetura.
    ```bash
    python benchmark_runner.py
    ```
    Este script irá rodar 100 partidas para cada tipo de agente (pode ser alterado no código) e, ao final, imprimirá uma tabela de resultados e salvará o gráfico `benchmark_comparativo.png`.

2.  **Torneio de IAs:** Para uma batalha direta com limite de tempo por jogada.
    ```bash
    python tournament.py
    ```
    Este script simula um torneio eliminatório entre os agentes e coroa um campeão.

## Resultados e Análise

Após a execução do `benchmark_runner.py` (com 50.000 simulações por decisão), os resultados de performance foram os seguintes:

| Arquitetura | Tempo Médio (s) | Aceleração (Speedup) |
| :--- | :---: | :---: |
| **CPU Single-Core** | ~207.5 s | 1x |
| **CPU Multi-Core** | ~73.0 s | **~2.8x** |
| **GPU (Numba)** | **~0.3 s** | **~719x** |

![Gráfico Comparativo](benchmark_comparativo.png)

### Conclusão Principal

[cite_start]Os resultados demonstram o imenso potencial da aceleração por GPU para cargas de trabalho massivamente paralelas, alcançando um speedup de mais de 700x em relação à execução sequencial. 

No entanto, o **Torneio de IAs** com limite de tempo revelou uma descoberta mais profunda: **performance bruta não se traduz diretamente em melhor estratégia se o algoritmo não for adequado à arquitetura**.

* Os agentes **CPU** (single e multi-core) tiveram um desempenho superior no torneio. Com baixo "overhead" por iteração, eles conseguiram executar muitos ciclos do MCTS, construindo uma árvore de decisão ampla e explorando diversas estratégias dentro do tempo limite.
* O agente **GPU**, apesar de ser mais rápido na tarefa de *rollout*, possui um alto custo de comunicação CPU-GPU. Dentro do limite de tempo, ele conseguia executar poucas iterações do MCTS, resultando em uma árvore de decisão "pobre" e jogadas estrategicamente inferiores.

[cite_start]A lição fundamental deste projeto é que a escolha da ferramenta de paralelização (CPU multi-core vs. GPU) depende criticamente da estrutura do algoritmo.  O MCTS, em sua forma clássica, beneficia-se mais da agilidade e baixa latência da CPU para construir sua árvore de forma iterativa.

## Trabalhos Futuros

* [cite_start]**Otimização de Kernel:** Aprofundar a otimização do kernel CUDA para reduzir ainda mais o tempo de rollout. 
* **Batching para GPU:** Modificar o algoritmo MCTS para acumular vários nós folha e enviá-los em um único "lote" (batch) para a GPU, minimizando o overhead de comunicação.
* [cite_start]**Computação Distribuída:** Implementar a versão com PySpark para problemas que excedam a capacidade de uma única máquina. 
* [cite_start]**Melhoria da IA:** Substituir os *rollouts* aleatórios por uma política heurística mais informada para melhorar a qualidade das simulações.