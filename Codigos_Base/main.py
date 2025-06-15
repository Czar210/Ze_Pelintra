import random
import time
import copy
from logica import JogoTruco2v2
from agente_mcts import MCTSAgente

def main():
    # Voltamos ao modo de benchmark silencioso
    start_time = time.time()
    bot_team1 = MCTSAgente(n_simulacoes=20000) 
    jogo = JogoTruco2v2(simulacao=True)

    JOGADOR_BOT_T1_ID = 1
    jogador_bot_t1_obj = jogo.jogadores[JOGADOR_BOT_T1_ID - 1]
    TIME_BOT_T1_ID = jogador_bot_t1_obj.time_id
    
    ultima_previsao_t1 = None
    
    while jogo.estado_jogo != "JOGO_FINALIZADO":
        
        if jogo.estado_jogo == "MAO_FINALIZADA":
            if ultima_previsao_t1 is not None:
                resultado_real = 1 if jogo.vencedor_mao == TIME_BOT_T1_ID else 0
                bot_team1.registrar_resultado_da_mao(ultima_previsao_t1, resultado_real)
                ultima_previsao_t1 = None

        estado_atual = jogo.estado_jogo

        if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]:
            jogo.iniciar_nova_mao()
            continue # Continua para o próximo loop para reavaliar o estado

        # ### LÓGICA DA MÃO DE ONZE CORRIGIDA ###
        elif estado_atual == "MAO_DE_ONZE":
            time_em_risco = 1 if jogo.pontos_time1 >= 11 else 2
            
            # 1. Distribui as cartas para que a decisão possa ser tomada
            jogo.distribuir_cartas()

            aceitou = False
            if time_em_risco == TIME_BOT_T1_ID:
                aceitou = bot_team1.decidir_mao_de_onze_com_mc(jogo, jogador_bot_t1_obj)
            else: # Time adversário decide aleatoriamente
                aceitou = random.choice([True, False])

            # 2. Reage à decisão
            if not aceitou: # Se decidiram CORRER
                # Anula a distribuição de cartas, dá o ponto de penalidade e finaliza a mão
                time_adversario = 2 if time_em_risco == 1 else 1
                jogo._dar_pontos(time_adversario, 1)
                jogo.estado_jogo = "MAO_FINALIZADA"
            
            # Se 'aceitou' for True, não fazemos nada, pois o jogo já está "EM_ANDAMENTO"
            # e pronto para continuar no próximo ciclo do loop.
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
    tempo_execucao = end_time - start_time
    placar_final = f"Time 1 ({jogo.pontos_time1}) vs Time 2 ({jogo.pontos_time2})"
    precisao_t1_mse = bot_team1.calcular_precisao_mse()

    print("\n" + "="*30)
    print("      RESULTADO DO BENCHMARK (MCTS)")
    print("="*30)
    print(f"Tempo de Execução : {tempo_execucao:.4f} segundos")
    print(f"Placar Final      : {placar_final}")
    print(f"Precisão Bot T1 (MSE) : {precisao_t1_mse:.4f} (menor = melhor)")
    print("="*30)

if __name__ == '__main__':
    main()