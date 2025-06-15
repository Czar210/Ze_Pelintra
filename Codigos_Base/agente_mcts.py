import random
import math
import copy
import numpy as np
from logica import JogoTruco2v2

# A classe MCTSNode permanece a mesma
class MCTSNode:
    """ Representa um nó na árvore de busca do Monte Carlo. """
    def __init__(self, estado_jogo, parente=None, jogada=None):
        self.estado_jogo = estado_jogo
        self.parente = parente
        self.jogada = jogada
        self.filhos = []
        self.vitorias = 0
        self.visitas = 0
        self.jogadas_nao_exploradas = self.estado_jogo.jogadores[self.estado_jogo.jogador_atual_idx].mao[:]

    def selecionar_filho_ucb(self):
        """ Seleciona o melhor filho usando a fórmula UCB1. """
        C = math.sqrt(2)
        log_visitas_pai = math.log(self.visitas)
        
        melhor_score = -1
        melhor_filho = None
        for filho in self.filhos:
            epsilon = 1e-6
            ucb_score = (filho.vitorias / (filho.visitas + epsilon)) + C * math.sqrt(log_visitas_pai / (filho.visitas + epsilon))
            if ucb_score > melhor_score:
                melhor_score = ucb_score
                melhor_filho = filho
        return melhor_filho

    def expandir(self):
        """ Expande a árvore criando um novo nó filho. """
        jogada = self.jogadas_nao_exploradas.pop()
        novo_estado = copy.deepcopy(self.estado_jogo)
        
        jogador_id = novo_estado.jogadores[novo_estado.jogador_atual_idx].id
        novo_estado.jogar_carta(jogador_id, jogada)
        
        filho = MCTSNode(estado_jogo=novo_estado, parente=self, jogada=jogada)
        self.filhos.append(filho)
        return filho

    def retropropagar(self, resultado):
        """ Atualiza as estatísticas de vitórias/visitas de volta até a raiz. """
        no_atual = self
        while no_atual is not None:
            no_atual.visitas += 1
            no_atual.vitorias += resultado
            no_atual = no_atual.parente

class MCTSAgente:
    """ O agente que usa MCTS para tomar decisões. """
    def __init__(self, n_simulacoes=1000):
        self.n_simulacoes = n_simulacoes
        self.log_previsoes = []

    def _simular_rollout(self, estado_jogo, time_bot_id):
        jogo_simulado = copy.deepcopy(estado_jogo)
        jogo_simulado.simulacao = True
        while jogo_simulado.estado_jogo == "EM_ANDAMENTO":
            jogador_da_vez = jogo_simulado.jogadores[jogo_simulado.jogador_atual_idx]
            if not jogador_da_vez.mao:
                jogo_simulado._checar_vencedor_da_mao()
                continue
            carta_aleatoria = random.choice(jogador_da_vez.mao)
            jogo_simulado.jogar_carta(jogador_da_vez.id, carta_aleatoria)
        return 1 if jogo_simulado.vencedor_mao == time_bot_id else 0

    # ### MÉTODO ATUALIZADO com a barra de progresso ###
    def decidir_melhor_jogada(self, estado_jogo, jogador_bot):
        """ Executa o algoritmo MCTS e retorna a melhor jogada. """
        time_bot_id = jogador_bot.time_id
        raiz = MCTSNode(estado_jogo=estado_jogo)

        if not raiz.jogadas_nao_exploradas:
            return None, 0.0

        # Loop principal do MCTS
        for i in range(self.n_simulacoes):
            no_atual = raiz
            
            # 1. Seleção
            while not no_atual.jogadas_nao_exploradas and no_atual.filhos:
                no_atual = no_atual.selecionar_filho_ucb()

            # 2. Expansão
            if no_atual.jogadas_nao_exploradas:
                no_atual = no_atual.expandir()

            # 3. Simulação (Rollout)
            if no_atual is not None:
                resultado_rollout = self._simular_rollout(no_atual.estado_jogo, time_bot_id)
                # 4. Retropropagação
                no_atual.retropropagar(resultado_rollout)

            # Lógica da Barra de Progresso
            # A cada 2% de progresso (ou na última iteração), atualiza a barra
            if (i + 1) % (self.n_simulacoes // 50) == 0 or (i + 1) == self.n_simulacoes:
                percentual = (i + 1) / self.n_simulacoes
                tamanho_barra = 30
                blocos_cheios = int(tamanho_barra * percentual)
                barra = "█" * blocos_cheios + "░" * (tamanho_barra - blocos_cheios)
                print(f"\rAnalisando... [{barra}] {percentual:.1%}", end="", flush=True)
        
        print() # Pula uma linha após a conclusão da barra

        if not raiz.filhos:
             return random.choice(estado_jogo.jogadores[estado_jogo.jogador_atual_idx].mao), 0.5

        melhor_filho = max(raiz.filhos, key=lambda c: c.visitas)
        taxa_vitoria_estimada = melhor_filho.vitorias / melhor_filho.visitas if melhor_filho.visitas > 0 else 0.0
        
        return melhor_filho.jogada, taxa_vitoria_estimada
    
    # Os outros métodos (registrar_resultado_da_mao, calcular_precisao_mse, etc.) permanecem os mesmos.
    def registrar_resultado_da_mao(self, previsao, resultado_real):
        if previsao is not None:
            self.log_previsoes.append((previsao, resultado_real))

    def calcular_precisao_mse(self):
        if not self.log_previsoes:
            return 0.0
        previsoes = np.array([p[0] for p in self.log_previsoes])
        resultados_reais = np.array([p[1] for p in self.log_previsoes])
        return np.mean((previsoes - resultados_reais)**2)

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

    def decidir_mao_de_onze_com_mc(self, estado_jogo_inicial, jogador_bot, n_simulacoes_mao_onze=200):
        """ Usa Monte Carlo para decidir se joga ou corre na Mão de Onze. """
        time_bot_id = jogador_bot.time_id
        
        # Cenário A: JOGAR a mão (o estado recebido já tem as cartas)
        vitorias_se_jogar = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_A = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_A.simulacao = True
            
            # REMOVIDO: jogo_para_simular_A.distribuir_cartas() -> Não é mais necessário aqui
            
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_A)
            if time_vencedor_partida == time_bot_id:
                vitorias_se_jogar += 1
        taxa_vitoria_jogando = vitorias_se_jogar / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0

        # Cenário B: CORRER da mão
        vitorias_se_correr = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_B = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_B.simulacao = True
            
            # Para simular o 'correr', voltamos o estado para antes de dar as cartas
            # e aplicamos a penalidade.
            time_adversario = 2 if time_bot_id == 1 else 1
            # Resetamos os pontos que o time do bot possa ter ganho na simulação de "jogar"
            if time_bot_id == 1: jogo_para_simular_B.pontos_time1 = estado_jogo_inicial.pontos_time1
            else: jogo_para_simular_B.pontos_time2 = estado_jogo_inicial.pontos_time2

            jogo_para_simular_B._dar_pontos(time_adversario, 1)
            jogo_para_simular_B.estado_jogo = "MAO_FINALIZADA"
            
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_B)
            if time_vencedor_partida == time_bot_id:
                vitorias_se_correr += 1
        taxa_vitoria_correndo = vitorias_se_correr / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0

        return taxa_vitoria_jogando >= taxa_vitoria_correndo