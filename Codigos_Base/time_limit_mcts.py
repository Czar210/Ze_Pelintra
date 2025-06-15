import random
import math
import copy
import numpy as np
import os
import time # <<< Importar time
from collections import Counter
from joblib import Parallel, delayed
from logica import JogoTruco2v2

# MCTSNode não muda
class MCTSNode:
    # ... (código inalterado) ...
    def __init__(self, estado_jogo, parente=None, jogada=None):
        self.estado_jogo = estado_jogo
        self.parente = parente
        self.jogada = jogada
        self.filhos = []
        self.vitorias = 0
        self.visitas = 0
        self.jogadas_nao_exploradas = self.estado_jogo.jogadores[self.estado_jogo.jogador_atual_idx].mao[:]
    def selecionar_filho_ucb(self):
        C = math.sqrt(2); log_visitas_pai = math.log(self.visitas)
        melhor_score = -1; melhor_filho = None
        for filho in self.filhos:
            epsilon = 1e-6
            ucb_score = (filho.vitorias / (filho.visitas + epsilon)) + C * math.sqrt(log_visitas_pai / (filho.visitas + epsilon))
            if ucb_score > melhor_score: melhor_score = ucb_score; melhor_filho = filho
        return melhor_filho
    def expandir(self):
        jogada = self.jogadas_nao_exploradas.pop(); novo_estado = copy.deepcopy(self.estado_jogo)
        jogador_id = novo_estado.jogadores[novo_estado.jogador_atual_idx].id
        novo_estado.jogar_carta(jogador_id, jogada)
        filho = MCTSNode(estado_jogo=novo_estado, parente=self, jogada=jogada); self.filhos.append(filho)
        return filho
    def retropropagar(self, resultado):
        no_atual = self
        while no_atual is not None:
            no_atual.visitas += 1; no_atual.vitorias += resultado; no_atual = no_atual.parente

# ### ATUALIZADO: Função de trabalho agora usa limite de tempo ###
def run_single_mcts_search_timed(estado_jogo, jogador_bot, time_limit):
    """
    Executa uma busca MCTS independente pelo tempo determinado.
    """
    start_time = time.time()
    sims_realizadas = 0
    
    agente_temporario = MCTSAgente() # Apenas para acessar o _simular_rollout
    time_bot_id = jogador_bot.time_id
    raiz = MCTSNode(estado_jogo=estado_jogo)

    if not raiz.jogadas_nao_exploradas:
        return None, 0.0, 0

    # O loop agora é baseado em tempo
    while time.time() - start_time < time_limit:
        no_atual = raiz
        while not no_atual.jogadas_nao_exploradas and no_atual.filhos:
            no_atual = no_atual.selecionar_filho_ucb()
        if no_atual.jogadas_nao_exploradas:
            no_atual = no_atual.expandir()
        if no_atual is not None:
            resultado_rollout = agente_temporario._simular_rollout(no_atual.estado_jogo, time_bot_id)
            no_atual.retropropagar(resultado_rollout)
        sims_realizadas += 1

    if not raiz.filhos:
        return random.choice(estado_jogo.jogadores[estado_jogo.jogador_atual_idx].mao), 0.5, sims_realizadas
    
    melhor_filho = max(raiz.filhos, key=lambda c: c.visitas)
    taxa_vitoria_estimada = melhor_filho.vitorias / melhor_filho.visitas if melhor_filho.visitas > 0 else 0.0
    
    # Retorna também o número de simulações que conseguiu fazer
    return melhor_filho.jogada, taxa_vitoria_estimada, sims_realizadas

class MCTSAgente:
    # ### ATUALIZADO: __init__ agora recebe time_limit ###
    def __init__(self, time_limit_por_jogada=1.0, n_jobs=-1):
        self.time_limit = time_limit_por_jogada
        self.log_previsoes = []
        self.n_jobs = n_jobs

    # ### ATUALIZADO: Orquestração paralela de workers baseados em tempo ###
    def decidir_melhor_jogada(self, estado_jogo, jogador_bot):
        if not jogador_bot.mao:
            return None, 0.0

        n_cores = os.cpu_count() or 1 if self.n_jobs == -1 else self.n_jobs
        
        # Cada núcleo rodará pelo tempo limite
        resultados_paralelos = Parallel(n_jobs=self.n_jobs)(
            delayed(run_single_mcts_search_timed)(copy.deepcopy(estado_jogo), jogador_bot, self.time_limit) for _ in range(n_cores)
        )

        jogadas_recomendadas = [res[0] for res in resultados_paralelos if res and res[0]]
        total_sims_realizadas = sum(res[2] for res in resultados_paralelos if res)
        
        print(f"    > {self.__class__.__name__} ({n_cores} núcleos) pensou por ~{self.time_limit:.1f}s e realizou {total_sims_realizadas} simulações.")
        
        if not jogadas_recomendadas:
            return random.choice(jogador_bot.mao), 0.5

        votos = Counter(jogadas_recomendadas)
        melhor_jogada = votos.most_common(1)[0][0]
        taxa_vitoria_estimada = next((res[1] for res in resultados_paralelos if res and res[0] == melhor_jogada), 0.5)

        return melhor_jogada, taxa_vitoria_estimada
        
    # O resto da classe (métodos de simulação e logging) permanece igual
    def _simular_rollout(self, estado_jogo, time_bot_id):
        # ... (código inalterado)
        jogo_simulado = copy.deepcopy(estado_jogo); jogo_simulado.simulacao = True
        while jogo_simulado.estado_jogo == "EM_ANDAMENTO":
            jogador_da_vez = jogo_simulado.jogadores[jogo_simulado.jogador_atual_idx]
            if not jogador_da_vez.mao: jogo_simulado._checar_vencedor_da_mao(); continue
            carta_aleatoria = random.choice(jogador_da_vez.mao)
            jogo_simulado.jogar_carta(jogador_da_vez.id, carta_aleatoria)
        return 1 if jogo_simulado.vencedor_mao == time_bot_id else 0
    def registrar_resultado_da_mao(self, previsao, resultado_real):
        # ... (código inalterado)
        if previsao is not None: self.log_previsoes.append((previsao, resultado_real))
    def calcular_precisao_mse(self):
        # ... (código inalterado)
        if not self.log_previsoes: return 0.0
        previsoes = np.array([p[0] for p in self.log_previsoes]); resultados_reais = np.array([p[1] for p in self.log_previsoes])
        return np.mean((previsoes - resultados_reais)**2)
    def _simular_jogo_completo(self, estado_jogo):
        # ... (código inalterado)
        jogo_simulado = copy.deepcopy(estado_jogo); jogo_simulado.simulacao = True
        while jogo_simulado.estado_jogo != "JOGO_FINALIZADO":
            estado_atual = jogo_simulado.estado_jogo
            if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]: jogo_simulado.iniciar_nova_mao()
            elif estado_atual == "MAO_DE_ONZE":
                aceitou = random.choice([True, False]); time_em_risco = 1 if jogo_simulado.pontos_time1 >= 11 else 2
                if not aceitou: jogo_simulado._dar_pontos(2 if time_em_risco == 1 else 1, 1); jogo_simulado.estado_jogo = "MAO_FINALIZADA"
                else: jogo_simulado.distribuir_cartas()
            elif estado_atual == "EM_ANDAMENTO":
                jogador_da_vez = jogo_simulado.jogadores[jogo_simulado.jogador_atual_idx]
                if not jogador_da_vez.mao: jogo_simulado._checar_vencedor_da_mao(); continue
                carta_aleatoria = random.choice(jogador_da_vez.mao); jogo_simulado.jogar_carta(jogador_da_vez.id, carta_aleatoria)
        if jogo_simulado.pontos_time1 >= 12: return 1
        elif jogo_simulado.pontos_time2 >= 12: return 2
        else: return 0
    def decidir_mao_de_onze_com_mc(self, estado_jogo_inicial, jogador_bot, n_simulacoes_mao_onze=200):
        # ... (código inalterado)
        time_bot_id = jogador_bot.time_id; vitorias_se_jogar = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_A = copy.deepcopy(estado_jogo_inicial); jogo_para_simular_A.simulacao = True
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_A)
            if time_vencedor_partida == time_bot_id: vitorias_se_jogar += 1
        taxa_vitoria_jogando = vitorias_se_jogar / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0
        vitorias_se_correr = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_B = copy.deepcopy(estado_jogo_inicial); jogo_para_simular_B.simulacao = True
            time_adversario = 2 if time_bot_id == 1 else 1
            if time_bot_id == 1: jogo_para_simular_B.pontos_time1 = estado_jogo_inicial.pontos_time1
            else: jogo_para_simular_B.pontos_time2 = estado_jogo_inicial.pontos_time2
            jogo_para_simular_B._dar_pontos(time_adversario, 1); jogo_para_simular_B.estado_jogo = "MAO_FINALIZADA"
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_B)
            if time_vencedor_partida == time_bot_id: vitorias_se_correr += 1
        taxa_vitoria_correndo = vitorias_se_correr / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0
        return taxa_vitoria_jogando >= taxa_vitoria_correndo