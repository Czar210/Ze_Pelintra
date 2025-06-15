import random
import copy
import numpy as np # Usaremos numpy para o cálculo do MSE
from logica import JogoTruco2v2, Carta, Jogador

class MonteCarloBot:
    def __init__(self, n_simulacoes=1000):
        self.n_simulacoes = n_simulacoes
        # Lista para guardar tuplas de (previsão, resultado_real)
        self.log_previsoes = []

    def registrar_resultado_da_mao(self, previsao, resultado_real):
        """
        Registra a previsão feita e o resultado real da mão.
        Args:
            previsao (float): A probabilidade de vitória que o bot calculou.
            resultado_real (int): 1 se o time do bot venceu a mão, 0 caso contrário.
        """
        if previsao is not None:
            self.log_previsoes.append((previsao, resultado_real))

    def calcular_precisao_mse(self):
        """
        Calcula o Erro Quadrático Médio (MSE) das previsões do bot.
        Retorna o MSE, ou 0 se nenhum registro foi feito.
        """
        if not self.log_previsoes:
            return 0.0
        
        previsoes = np.array([p[0] for p in self.log_previsoes])
        resultados_reais = np.array([p[1] for p in self.log_previsoes])
        
        mse = np.mean((previsoes - resultados_reais)**2)
        return mse

    def decidir_melhor_jogada(self, estado_jogo_atual, jogador_bot):
        """
        Avalia todas as jogadas e retorna a melhor carta E a sua taxa de vitória.
        """
        jogadas_possiveis = jogador_bot.mao[:]
        if not jogadas_possiveis:
            return None, None

        maior_taxa_vitoria = -1.0
        melhor_jogada = None
        
        for jogada in jogadas_possiveis:
            vitorias = 0
            for _ in range(self.n_simulacoes):
                estado_copia = copy.deepcopy(estado_jogo_atual)
                jogador_bot_copia = next(p for p in estado_copia.jogadores if p.id == jogador_bot.id)
                carta_para_jogar = next(c for c in jogador_bot_copia.mao if c == jogada)
                estado_copia.jogar_carta(jogador_bot_copia.id, carta_para_jogar)
                vitorias += self._determinize_and_simulate(estado_copia, jogador_bot.id)
            
            taxa_vitoria = vitorias / self.n_simulacoes
            
            if taxa_vitoria > maior_taxa_vitoria:
                maior_taxa_vitoria = taxa_vitoria
                melhor_jogada = jogada

        # Retorna a jogada e a previsão de vitória
        return melhor_jogada, maior_taxa_vitoria
        
    def decidir_mao_de_onze_com_mc(self, estado_jogo_inicial, jogador_bot, n_simulacoes_mao_onze=200):
        # A lógica interna permanece a mesma, mas sem os prints
        time_bot_id = jogador_bot.time_id
        
        vitorias_se_jogar = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_A = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_A.simulacao = True
            jogo_para_simular_A.distribuir_cartas()
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_A)
            if time_vencedor_partida == time_bot_id:
                vitorias_se_jogar += 1
        taxa_vitoria_jogando = vitorias_se_jogar / n_simulacoes_mao_onze

        vitorias_se_correr = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_B = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_B.simulacao = True
            time_adversario = 2 if time_bot_id == 1 else 1
            jogo_para_simular_B._dar_pontos(time_adversario, 1)
            jogo_para_simular_B.estado_jogo = "MAO_FINALIZADA"
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_B)
            if time_vencedor_partida == time_bot_id:
                vitorias_se_correr += 1
        taxa_vitoria_correndo = vitorias_se_correr / n_simulacoes_mao_onze

        return taxa_vitoria_jogando >= taxa_vitoria_correndo

    # As funções _run_single_simulation, _determinize_and_simulate e _simular_jogo_completo
    # continuam as mesmas, pois já são silenciosas.
    def _run_single_simulation(self, estado_jogo_determinizado):
        jogo_simulado = estado_jogo_determinizado
        while jogo_simulado.estado_jogo == "EM_ANDAMENTO":
            jogador_da_vez = jogo_simulado.jogadores[jogo_simulado.jogador_atual_idx]
            if not jogador_da_vez.mao:
                jogo_simulado._checar_vencedor_da_mao(); continue
            carta_aleatoria = random.choice(jogador_da_vez.mao)
            jogo_simulado.jogar_carta(jogador_da_vez.id, carta_aleatoria)
        return jogo_simulado.vencedor_mao

    def _determinize_and_simulate(self, estado_jogo_copia, bot_player_id):
        bot_player = next((p for p in estado_jogo_copia.jogadores if p.id == bot_player_id), None)
        cartas_conhecidas = set(carta for p in estado_jogo_copia.jogadores if p.id == bot_player_id for carta in p.mao)
        cartas_conhecidas.update(c for _, c in estado_jogo_copia.cartas_na_mesa)
        if estado_jogo_copia.vira: cartas_conhecidas.add(estado_jogo_copia.vira)
        cartas_desconhecidas = [c for c in estado_jogo_copia.baralho_completo if c not in cartas_conhecidas]
        random.shuffle(cartas_desconhecidas)
        for p in estado_jogo_copia.jogadores:
            if p.id != bot_player_id:
                for _ in range(3 - len(p.mao)):
                    if cartas_desconhecidas: p.mao.append(cartas_desconhecidas.pop())
        vencedor_time_id = self._run_single_simulation(estado_jogo_copia)
        return 1 if bot_player and vencedor_time_id == bot_player.time_id else 0

    def _simular_jogo_completo(self, estado_jogo):
        jogo_simulado = estado_jogo
        while jogo_simulado.estado_jogo != "JOGO_FINALIZADO":
            estado_atual = jogo_simulado.estado_jogo
            if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]:
                jogo_simulado.iniciar_nova_mao()
            elif estado_atual == "MAO_DE_ONZE":
                aceitou = random.choice([True, False])
                time_em_risco = 1 if jogo_simulado.pontos_time1 >= 11 else 2
                if not aceitou:
                    jogo_simulado._dar_pontos(2 if time_em_risco == 1 else 1, 1)
                    jogo_simulado.estado_jogo = "MAO_FINALIZADA"
                else:
                    jogo_simulado.distribuir_cartas()
            elif estado_atual == "EM_ANDAMENTO":
                jogador_da_vez = jogo_simulado.jogadores[jogo_simulado.jogador_atual_idx]
                if not jogador_da_vez.mao:
                    jogo_simulado._checar_vencedor_da_mao(); continue
                carta_aleatoria = random.choice(jogador_da_vez.mao)
                jogo_simulado.jogar_carta(jogador_da_vez.id, carta_aleatoria)
        if jogo_simulado.pontos_time1 >= 12: return 1
        elif jogo_simulado.pontos_time2 >= 12: return 2
        else: return 0