import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import copy

# Importa as classes dos nossos diferentes agentes e a lógica do jogo
from logica import JogoTruco2v2
from agente_mcts_multi import MCTSAgente as AgenteCPU  # Agente de CPU para single e multi-core
from agente_gpu import GPUAgenteMCTS as AgenteGPU

def run_single_game(tipo_agente, n_simulacoes_mcts):
    """
    Executa uma única partida de Truco do início ao fim e retorna um dicionário com as métricas.
    """
    # 1. Instancia o agente correto baseado no tipo
    if tipo_agente == 'single':
        bot_team1 = AgenteCPU(n_simulacoes=n_simulacoes_mcts, n_jobs=1)
    elif tipo_agente == 'multi':
        bot_team1 = AgenteCPU(n_simulacoes=n_simulacoes_mcts, n_jobs=-1)
    else: # 'gpu'
        # Convertemos o número total de simulações para "passos" do MCTS na GPU
        n_mcts_steps = n_simulacoes_mcts // 4096 
        if n_mcts_steps == 0: n_mcts_steps = 1 # Garante pelo menos 1 passo
        bot_team1 = AgenteGPU(n_simulacoes=n_mcts_steps)
    
    jogo = JogoTruco2v2(simulacao=True)
    
    # 2. Configura a partida
    start_time = time.time()
    
    JOGADOR_BOT_T1_ID = 1
    jogador_bot_t1_obj = jogo.jogadores[JOGADOR_BOT_T1_ID - 1]
    TIME_BOT_T1_ID = jogador_bot_t1_obj.time_id
    ultima_previsao_t1 = None
    
    # 3. Loop principal do jogo (lógica de main.py)
    while jogo.estado_jogo != "JOGO_FINALIZADO":
        if jogo.estado_jogo == "MAO_FINALIZADA":
            if ultima_previsao_t1 is not None:
                resultado_real = 1 if jogo.vencedor_mao == TIME_BOT_T1_ID else 0
                bot_team1.registrar_resultado_da_mao(ultima_previsao_t1, resultado_real)
                ultima_previsao_t1 = None
        estado_atual = jogo.estado_jogo
        if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]:
            jogo.iniciar_nova_mao()
            continue
        elif estado_atual == "MAO_DE_ONZE":
            jogo.distribuir_cartas()
            aceitou = False
            if TIME_BOT_T1_ID == (1 if jogo.pontos_time1 >= 11 else 2):
                aceitou = bot_team1.decidir_mao_de_onze_com_mc(jogo, jogador_bot_t1_obj)
            else:
                aceitou = random.choice([True, False])
            if not aceitou:
                time_adversario = 2 if TIME_BOT_T1_ID == 1 else 1
                jogo._dar_pontos(time_adversario, 1)
                jogo.estado_jogo = "MAO_FINALIZADA"
            continue
        elif estado_atual == "EM_ANDAMENTO":
            jogador_da_vez = jogo.jogadores[jogo.jogador_atual_idx]
            if not jogador_da_vez.mao:
                jogo._checar_vencedor_da_mao()
                if jogo.estado_jogo == "JOGO_FINALIZADO": break
                continue
            carta_jogada = None
            if jogador_da_vez.id == JOGADOR_BOT_T1_ID:
                carta_jogada, ultima_previsao_t1 = bot_team1.decidir_melhor_jogada(jogo, jogador_da_vez)
            else:
                carta_jogada = random.choice(jogador_da_vez.mao)
            if carta_jogada:
                jogo.jogar_carta(jogador_da_vez.id, carta_jogada)
            if jogo.estado_jogo == "JOGO_FINALIZADO":
                if ultima_previsao_t1 is not None:
                    resultado_real = 1 if jogo.vencedor_mao == TIME_BOT_T1_ID else 0
                    bot_team1.registrar_resultado_da_mao(ultima_previsao_t1, resultado_real)
                break
    
    end_time = time.time()

    # 4. Coleta e retorna as métricas da partida
    tempo_total = end_time - start_time
    venceu = 1 if jogo.pontos_time1 >= 12 else 0
    return {
        "tipo_agente": tipo_agente,
        "tempo_execucao": tempo_total,
        "pontos_feitos": jogo.pontos_time1,
        "pontos_tomados": jogo.pontos_time2,
        "total_maos": jogo.mao_atual,
        "vitoria": venceu,
        "precisao_mse": bot_team1.calcular_precisao_mse()
    }

def main():
    # NOTA: Para um teste rápido, comece com N_PARTIDAS = 5.
    # Para o resultado final, use N_PARTIDAS = 100.
    N_PARTIDAS = 5
    N_SIMULACOES = 50000

    todos_os_resultados = []
    tipos_de_agente = ['single', 'multi', 'gpu']

    for agente_tipo in tipos_de_agente:
        print(f"\n{'='*40}\nINICIANDO BENCHMARK PARA O AGENTE: {agente_tipo.upper()}\n{'='*40}")
        for i in range(N_PARTIDAS):
            print(f"  -> Rodando partida {i+1}/{N_PARTIDAS}...")
            resultado = run_single_game(agente_tipo, N_SIMULACOES)
            todos_os_resultados.append(resultado)
            print(f"     ...concluído em {resultado['tempo_execucao']:.2f}s. Placar: {resultado['pontos_feitos']} a {resultado['pontos_tomados']}.")

    df = pd.DataFrame(todos_os_resultados)
    df_summary = df.groupby('tipo_agente').agg(
        tempo_medio=('tempo_execucao', 'mean'),
        winrate=('vitoria', 'mean'),
        maos_por_partida=('total_maos', 'mean'),
        pontos_feitos_medio=('pontos_feitos', 'mean'),
        pontos_tomados_medio=('pontos_tomados', 'mean'),
        mse_medio=('precisao_mse', 'mean')
    ).reindex(['single', 'multi', 'gpu']) # Garante a ordem no gráfico

    df_summary['winrate'] = df_summary['winrate'] * 100

    print("\n\n" + "="*50)
    print("      RESULTADO FINAL DO BENCHMARK (MÉDIAS)")
    print("="*50)
    print(df_summary.to_string())
    print("="*50)

    # Geração do Gráfico
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    g = sns.barplot(data=df_summary, x='tipo_agente', y='tempo_medio', ax=ax1, palette='viridis')
    ax1.set_ylabel('Tempo Médio de Execução (s)', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_title('Análise Comparativa de Performance dos Agentes (Média de {} Partidas)'.format(N_PARTIDAS), fontsize=16, pad=20)
    ax1.set_xlabel('Tipo de Agente', fontsize=14)
    
    for p in g.patches:
        ax1.annotate(f"{p.get_height():.2f}s", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=12, color='white', xytext=(0, -15),
                     textcoords='offset points', weight='bold')

    ax2 = ax1.twinx()
    sns.lineplot(data=df_summary, x='tipo_agente', y='winrate', ax=ax2, color='r', marker='o', lw=3, label='Winrate (%)')
    ax2.set_ylabel('Taxa de Vitória (%)', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    plt.savefig('benchmark_comparativo.png')
    print("\nGráfico 'benchmark_comparativo.png' salvo no diretório.")
    plt.show()

if __name__ == '__main__':
    main()