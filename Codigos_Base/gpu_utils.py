import numpy as np
import random
from logica import Carta, JogoTruco2v2

def criar_mapeamento_cartas():
    # ... (código da função anterior)
    ranks = ['4', '5', '6', '7', 'Q', 'J', 'K', 'A', '2', '3']
    naipes = ['Ouros', 'Espadas', 'Copas', 'Paus']
    carta_para_int = {}
    int_para_carta = {}
    i = 0
    for rank in ranks:
        for naipe in naipes:
            carta = Carta(rank, naipe)
            carta_para_int[carta] = i
            int_para_carta[i] = carta
            i += 1
    return carta_para_int, int_para_carta

CARTA_PARA_INT, INT_PARA_CARTA = criar_mapeamento_cartas()


def achatar_estado_para_gpu(jogo_atual, bot_id, n_simulacoes):
    """
    Cria N cenários hipotéticos (determinizações) e os converte em arrays NumPy.
    """
    # 1. Prepara os arrays NumPy vazios que serão preenchidos
    maos_array = np.zeros((n_simulacoes, 4, 3), dtype=np.int32)
    viras_array = np.zeros(n_simulacoes, dtype=np.int32)
    jogadores_iniciais_array = np.zeros(n_simulacoes, dtype=np.int32)
    
    # Identifica as cartas conhecidas no estado ATUAL do jogo
    cartas_conhecidas = set()
    bot_mao_original = []
    
    for p in jogo_atual.jogadores:
        if p.id == bot_id:
            for c in p.mao:
                cartas_conhecidas.add(c)
                bot_mao_original.append(c) # Guarda a mão original do bot
            break

    for _, carta_mesa in jogo_atual.cartas_na_mesa:
        cartas_conhecidas.add(carta_mesa)
    
    if jogo_atual.vira:
        cartas_conhecidas.add(jogo_atual.vira)

    baralho_completo = [c for c in CARTA_PARA_INT.keys()]

    # 2. Loop para criar N cenários diferentes
    for i in range(n_simulacoes):
        # Para cada simulação, cria um novo baralho de cartas desconhecidas
        cartas_desconhecidas = [c for c in baralho_completo if c not in cartas_conhecidas]
        random.shuffle(cartas_desconhecidas)
        
        # Preenche as mãos dos oponentes
        maos_hipoteticas = [[] for _ in range(4)]
        maos_hipoteticas[bot_id - 1] = bot_mao_original[:] # Mão do bot é fixa
        
        for j, p in enumerate(jogo_atual.jogadores):
            if p.id != bot_id:
                # O número de cartas a distribuir depende de quantas ele já jogou
                num_cartas_a_dar = len(bot_mao_original) - len(p.mao)
                maos_hipoteticas[j] = p.mao[:] + [cartas_desconhecidas.pop() for _ in range(num_cartas_a_dar)]

        # Converte o cenário hipotético para inteiros e preenche os arrays
        for j in range(4): # Para cada jogador
            for k in range(3): # Para cada carta
                if k < len(maos_hipoteticas[j]):
                    maos_array[i, j, k] = CARTA_PARA_INT[maos_hipoteticas[j][k]]
                else: # Se o jogador tiver menos de 3 cartas (já jogou algumas)
                    maos_array[i, j, k] = -1 # Usa -1 para indicar ausência de carta

        viras_array[i] = CARTA_PARA_INT[jogo_atual.vira]
        jogadores_iniciais_array[i] = jogo_atual.jogador_atual_idx

    return maos_array, viras_array, jogadores_iniciais_array