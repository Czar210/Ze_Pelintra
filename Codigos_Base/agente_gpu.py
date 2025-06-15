import numpy as np
from numba import cuda, types
import math
import random
import copy
from logica import JogoTruco2v2, Carta
from gpu_utils import achatar_estado_para_gpu 
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# ======================================================================
# Seção 1: Funções da GPU (Device e Kernel)
# ======================================================================

@cuda.jit(device=True)
def valor_da_carta_gpu(carta_int, rank_manilha):
    """Calcula o valor de uma carta para fins de comparação (roda na GPU)."""
    if carta_int == -1:
        return -1
    rank_carta = carta_int // 4
    if rank_carta == rank_manilha:
        naipe = carta_int % 4
        return 10 + (3 - naipe)
    else:
        return rank_carta

@cuda.jit
def simular_rollouts_gpu(
    maos_iniciais,
    viras,
    jogadores_iniciais,
    rng_states,
    resultados
):
    """Kernel CUDA: Simula uma mão completa de Truco para cada thread."""
    i = cuda.grid(1)
    if i >= maos_iniciais.shape[0]:
        return

    mao_thread = cuda.local.array((4, 3), dtype=types.int8)
    for j in range(4):
        for k in range(3):
            mao_thread[j, k] = maos_iniciais[i, j, k]

    vira = viras[i]
    jogador_atual = jogadores_iniciais[i]
    rank_manilha = (vira // 4 + 1) % 10
    vitorias_turno = cuda.local.array(3, dtype=types.int8)
    vencedor_primeiro_turno = -1
    
    for turno in range(3):
        mesa = cuda.local.array(4, dtype=types.int8)
        jogadores_na_mesa = cuda.local.array(4, dtype=types.int8)
        
        for j in range(4):
            jogador_idx = (jogador_atual + j) % 4
            cartas_validas_count = 0
            for k in range(3):
                if mao_thread[jogador_idx, k] != -1:
                    cartas_validas_count += 1
            
            if cartas_validas_count > 0:
                rand_float = xoroshiro128p_uniform_float32(rng_states, i)
                escolha_aleatoria = int(rand_float * cartas_validas_count)
                cartas_vistas = 0
                for k in range(3):
                    if mao_thread[jogador_idx, k] != -1:
                        if cartas_vistas == escolha_aleatoria:
                            mesa[j] = mao_thread[jogador_idx, k]
                            mao_thread[jogador_idx, k] = -1
                            break
                        cartas_vistas += 1
            else:
                mesa[j] = -1
            jogadores_na_mesa[j] = jogador_idx
        
        maior_valor = -1
        vencedor_temp_idx = -1
        for j in range(4):
            valor_carta = valor_da_carta_gpu(mesa[j], rank_manilha)
            if valor_carta > maior_valor:
                maior_valor = valor_carta
                vencedor_temp_idx = jogadores_na_mesa[j]
        empate = False
        if maior_valor > -1:
            contagem_maior_valor = 0
            for j in range(4):
                if valor_da_carta_gpu(mesa[j], rank_manilha) == maior_valor:
                    contagem_maior_valor += 1
            if contagem_maior_valor > 1:
                empate = True
        if empate:
            vitorias_turno[turno] = 0
        else:
            time_vencedor = (vencedor_temp_idx % 2) + 1
            vitorias_turno[turno] = time_vencedor
            jogador_atual = vencedor_temp_idx
            if turno == 0:
                vencedor_primeiro_turno = time_vencedor

    vitorias_time_1 = 0
    vitorias_time_2 = 0
    for t in range(3):
        if vitorias_turno[t] == 1: vitorias_time_1 += 1
        elif vitorias_turno[t] == 2: vitorias_time_2 += 1
    vencedor_final = 0
    if vitorias_time_1 == 1 and vitorias_time_2 == 0 and vitorias_turno[0] == 0:
        vencedor_final = 1
    elif vitorias_time_2 == 1 and vitorias_time_1 == 0 and vitorias_turno[0] == 0:
        vencedor_final = 2
    elif vitorias_time_1 >= 2:
        vencedor_final = 1
    elif vitorias_time_2 >= 2:
        vencedor_final = 2
    elif vitorias_time_1 == 1 and vitorias_time_2 == 1:
        if vitorias_turno[0] != 0:
            vencedor_final = vitorias_turno[0]

    if vencedor_final == 1:
        resultados[i] = 1
    else:
        resultados[i] = 0

# ======================================================================
# Seção 2: A Classe Principal do Agente GPU
# ======================================================================

class MCTSNode: # Classe Nó necessária para o MCTS na CPU
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
            if ucb_score > melhor_score:
                melhor_score = ucb_score; melhor_filho = filho
        return melhor_filho

    def expandir(self):
        jogada = self.jogadas_nao_exploradas.pop()
        novo_estado = copy.deepcopy(self.estado_jogo)
        jogador_id = novo_estado.jogadores[novo_estado.jogador_atual_idx].id
        novo_estado.jogar_carta(jogador_id, jogada)
        filho = MCTSNode(estado_jogo=novo_estado, parente=self, jogada=jogada)
        self.filhos.append(filho)
        return filho

    def retropropagar(self, resultado):
        no_atual = self
        while no_atual is not None:
            no_atual.visitas += 1
            no_atual.vitorias += resultado
            no_atual = no_atual.parente

class GPUAgenteMCTS:
    def __init__(self, n_simulacoes=20000):
        self.n_simulacoes = n_simulacoes
        self.n_rollouts_por_decisao = 4096
        self.log_previsoes = []

    # --- O Coração do MCTS (executado na CPU) ---
    def decidir_melhor_jogada(self, estado_jogo, jogador_bot):
        raiz = MCTSNode(estado_jogo=estado_jogo)
        if not raiz.jogadas_nao_exploradas:
            return None, 0.0

        # O MCTS roda um número fixo de vezes para construir a árvore
        for _ in range(self.n_simulacoes // self.n_rollouts_por_decisao):
            no_atual = raiz
            while not no_atual.jogadas_nao_exploradas and no_atual.filhos:
                no_atual = no_atual.selecionar_filho_ucb()
            if no_atual.jogadas_nao_exploradas:
                no_atual = no_atual.expandir()
            
            # A etapa de simulação agora é o rollout massivo na GPU
            taxa_vitoria = self._gpu_rollout(no_atual.estado_jogo, jogador_bot.id)
            
            no_atual.retropropagar(taxa_vitoria)

        if not raiz.filhos:
            return random.choice(jogador_bot.mao), 0.5

        melhor_filho = max(raiz.filhos, key=lambda c: c.visitas)
        taxa_vitoria_estimada = melhor_filho.vitorias / melhor_filho.visitas if melhor_filho.visitas > 0 else 0.0
        return melhor_filho.jogada, taxa_vitoria_estimada

    # --- O Orquestrador da GPU ---
    def _gpu_rollout(self, estado_jogo: JogoTruco2v2, bot_id: int):
        maos_iniciais, viras, jogadores_iniciais = achatar_estado_para_gpu(
            estado_jogo, bot_id, self.n_rollouts_por_decisao)
        rng_states = create_xoroshiro128p_states(self.n_rollouts_por_decisao, seed=random.randint(0, 2**32-1))

        d_maos = cuda.to_device(maos_iniciais)
        d_viras = cuda.to_device(viras)
        d_jogadores = cuda.to_device(jogadores_iniciais)
        d_rng_states = cuda.to_device(rng_states)
        d_resultados = cuda.device_array(self.n_rollouts_por_decisao, dtype=np.int8)

        threads_por_bloco = 128
        blocos_por_grid = math.ceil(self.n_rollouts_por_decisao / threads_por_bloco)
        
        simular_rollouts_gpu[blocos_por_grid, threads_por_bloco](
            d_maos, d_viras, d_jogadores, d_rng_states, d_resultados)
        cuda.synchronize() 

        resultados_host = d_resultados.copy_to_host()
        return np.mean(resultados_host)

    # --- Métodos de Benchmark e Decisões Estratégicas (CPU) ---
    def registrar_resultado_da_mao(self, previsao, resultado_real):
        if previsao is not None:
            self.log_previsoes.append((previsao, resultado_real))

    def calcular_precisao_mse(self):
        if not self.log_previsoes: return 0.0
        previsoes = np.array([p[0] for p in self.log_previsoes])
        resultados_reais = np.array([p[1] for p in self.log_previsoes])
        return np.mean((previsoes - resultados_reais)**2)

    def _simular_jogo_completo(self, estado_jogo):
        jogo_simulado = copy.deepcopy(estado_jogo)
        jogo_simulado.simulacao = True
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
        time_bot_id = jogador_bot.time_id
        vitorias_se_jogar = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_A = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_A.simulacao = True
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_A)
            if time_vencedor_partida == time_bot_id: vitorias_se_jogar += 1
        taxa_vitoria_jogando = vitorias_se_jogar / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0
        vitorias_se_correr = 0
        for _ in range(n_simulacoes_mao_onze):
            jogo_para_simular_B = copy.deepcopy(estado_jogo_inicial)
            jogo_para_simular_B.simulacao = True
            time_adversario = 2 if time_bot_id == 1 else 1
            if time_bot_id == 1: jogo_para_simular_B.pontos_time1 = estado_jogo_inicial.pontos_time1
            else: jogo_para_simular_B.pontos_time2 = estado_jogo_inicial.pontos_time2
            jogo_para_simular_B._dar_pontos(time_adversario, 1)
            jogo_para_simular_B.estado_jogo = "MAO_FINALIZADA"
            time_vencedor_partida = self._simular_jogo_completo(jogo_para_simular_B)
            if time_vencedor_partida == time_bot_id: vitorias_se_correr += 1
        taxa_vitoria_correndo = vitorias_se_correr / n_simulacoes_mao_onze if n_simulacoes_mao_onze > 0 else 0
        return taxa_vitoria_jogando >= taxa_vitoria_correndo