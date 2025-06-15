import random
import unittest
import copy

class Carta:
    """Representa uma carta do baralho."""
    RANKS = {'4': 0, '5': 1, '6': 2, '7': 3, 'Q': 4, 'J': 5, 'K': 6, 'A': 7, '2': 8, '3': 9}
    NAIPES = {'Ouros': 0, 'Espadas': 1, 'Copas': 2, 'Paus': 3}

    def __init__(self, rank, naipe):
        if rank not in self.RANKS:
            raise ValueError(f"Valor da carta inválido: {rank}")
        if naipe not in self.NAIPES:
            raise ValueError(f"Naipe inválido: {naipe}")
        self.rank = rank
        self.naipe = naipe
        self.valor_normal = self.RANKS[rank]

    def __str__(self):
        return f"{self.rank} de {self.naipe}"

    def __repr__(self):
        return f"Carta('{self.rank}', '{self.naipe}')"

    def __eq__(self, other):
        return self.rank == other.rank and self.naipe == other.naipe
    
    def __hash__(self):
        return hash((self.rank, self.naipe))

class Jogador:
    """Representa um jogador no jogo."""
    def __init__(self, id, time_id):
        self.id = id
        self.time_id = time_id
        self.mao = []

    def __repr__(self):
        return f"Jogador(id={self.id}, time={self.time_id})"

class JogoTruco2v2:
    """Gerencia a lógica e o estado de um jogo de Truco 2v2."""

    def __init__(self, simulacao=False):
        self.jogadores = [Jogador(1, 1), Jogador(2, 2), Jogador(3, 1), Jogador(4, 2)]
        self.baralho_completo = self._criar_baralho()
        
        self.pontos_time1 = 0
        self.pontos_time2 = 0
        self.jogador_iniciou_rodada_idx = -1
        self.mao_atual = 0
        
        self.estado_jogo = "NOVA_MAO"
        self.simulacao = simulacao 
        self.resetar_estado_da_mao()

    def _gerar_placar_visual(self):
        """Cria uma string de placar visual com uma barra de progresso."""
        # Barra de Progresso
        progresso_max = 12
        pontos_max = max(self.pontos_time1, self.pontos_time2)
        percentual = pontos_max / progresso_max if progresso_max > 0 else 0
        
        tamanho_barra = 20
        blocos_cheios = int(tamanho_barra * percentual)
        blocos_vazios = tamanho_barra - blocos_cheios
        
        barra = "█" * blocos_cheios + "░" * blocos_vazios
        
        # Placar
        t1_str = f"Time 1: {self.pontos_time1:02d}"
        t2_str = f"Time 2: {self.pontos_time2:02d}"
        
        return f"\n--- Mão {self.mao_atual} | {t1_str} vs {t2_str} | Progresso: [{barra}] {percentual:.0%} ---"

    def iniciar_nova_mao(self):
        self.mao_atual += 1
        if not self.simulacao:
            print(self._gerar_placar_visual())
        
        self.resetar_estado_da_mao()

        if self.pontos_time1 >= 11 and self.pontos_time2 < 11:
            self.estado_jogo = "MAO_DE_ONZE"
            self.valor_mao = 3
            if not self.simulacao: print("!!! MÃO DE ONZE PARA O TIME 1 !!!")
            return
        if self.pontos_time2 >= 11 and self.pontos_time1 < 11:
            self.estado_jogo = "MAO_DE_ONZE"
            self.valor_mao = 3
            if not self.simulacao: print("!!! MÃO DE ONZE PARA O TIME 2 !!!")
            return
        
        self.distribuir_cartas()
        
    def jogar_carta(self, jogador_id, carta):
        jogador = self.jogadores[self.jogador_atual_idx]
        if jogador.id != jogador_id:
            raise ValueError(f"Não é a vez do jogador {jogador_id}.")
        if carta not in jogador.mao:
            raise ValueError("Jogador não possui esta carta.")

        # Anúncio da jogada é feito pelo agente ou pelo main loop, não aqui.
        self.cartas_na_mesa.append((jogador, carta))
        jogador.mao.remove(carta)
        
        self.jogador_atual_idx = (self.jogador_atual_idx + 1) % 4

        if len(self.cartas_na_mesa) == 4:
            self._finalizar_turno()

    def _finalizar_turno(self):
        carta_mais_forte_tupla = self.cartas_na_mesa[0]
        valor_mais_alto = self.valor_da_carta(carta_mais_forte_tupla[1])
        
        for jogador_carta_tupla in self.cartas_na_mesa[1:]:
             valor_atual = self.valor_da_carta(jogador_carta_tupla[1])
             if valor_atual > valor_mais_alto:
                 valor_mais_alto = valor_atual
                 carta_mais_forte_tupla = jogador_carta_tupla

        valores = [self.valor_da_carta(c[1]) for c in self.cartas_na_mesa]
        if valores.count(valor_mais_alto) > 1:
            vencedor_turno_time = 0
            self.jogador_atual_idx = self.vencedor_turno_idx 
        else:
            vencedor_turno = carta_mais_forte_tupla[0]
            vencedor_turno_time = vencedor_turno.time_id
            self.jogador_atual_idx = self.jogadores.index(vencedor_turno)
            self.vencedor_turno_idx = self.jogador_atual_idx

        self.resultado_rodada[self.rodada_atual - 1] = vencedor_turno_time
        self.cartas_na_mesa = []
        self.rodada_atual += 1
        
        self._checar_vencedor_da_mao()

    def _finalizar_mao(self, time_vencedor):
        self.vencedor_mao = time_vencedor
        if time_vencedor in [1, 2]:
            if not self.simulacao:
                print(f"   -> Mão vencida pelo Time {time_vencedor}! (Valor: {self.valor_mao} ponto(s))")
            self._dar_pontos(time_vencedor, self.valor_mao)
        else:
             if not self.simulacao: print("   -> Mão empatada! Ninguém marca ponto.")

        self.estado_jogo = "MAO_FINALIZADA"
        if self.pontos_time1 >= 12 or self.pontos_time2 >= 12:
            self.estado_jogo = "JOGO_FINALIZADO"
            if not self.simulacao:
                print(self._gerar_placar_visual())
                print("\n" + "="*20 + " FIM DE JOGO " + "="*20)

    def _criar_baralho(self):
        return [Carta(r, n) for r in Carta.RANKS for n in Carta.NAIPES]

    def _definir_manilhas(self):
        ranks_ordenados = list(Carta.RANKS.keys())
        vira_rank_index = ranks_ordenados.index(self.vira.rank)
        manilha_rank = ranks_ordenados[(vira_rank_index + 1) % len(ranks_ordenados)]
        self.manilhas = {n: Carta(manilha_rank, n) for n in Carta.NAIPES}

    def valor_da_carta(self, carta):
        for naipe, manilha in self.manilhas.items():
            if carta.rank == manilha.rank and carta.naipe == manilha.naipe:
                return 10 + Carta.NAIPES[naipe]
        return carta.valor_normal

    def resolver_mao_de_onze(self, time_em_risco, aceitou):
        if aceitou:
            if not self.simulacao: print(f"   -> Time {time_em_risco} decidiu jogar a Mão de Onze!")
            self.distribuir_cartas()
        else:
            if not self.simulacao: print(f"   -> Time {time_em_risco} correu da Mão de Onze!")
            time_adversario = 2 if time_em_risco == 1 else 1
            self._dar_pontos(time_adversario, 1)
            self.estado_jogo = "MAO_FINALIZADA"

    def distribuir_cartas(self):
        self.estado_jogo = "EM_ANDAMENTO"
        baralho = self.baralho_completo[:]
        random.shuffle(baralho)
        for jogador in self.jogadores:
            jogador.mao = [baralho.pop() for _ in range(3)]
        self.vira = baralho.pop()
        self._definir_manilhas()
        self.jogador_iniciou_rodada_idx = (self.jogador_iniciou_rodada_idx + 1) % 4
        self.jogador_atual_idx = self.jogador_iniciou_rodada_idx
        self.vencedor_turno_idx = self.jogador_iniciou_rodada_idx
        if not self.simulacao:
            print(f"   Vira: {self.vira}. Jogador {self.jogadores[self.jogador_atual_idx].id} começa.")

    def _checar_vencedor_da_mao(self):
        vitorias_t1 = self.resultado_rodada.count(1)
        vitorias_t2 = self.resultado_rodada.count(2)
        empates = self.resultado_rodada.count(0)
        if vitorias_t1 >= 2: return self._finalizar_mao(1)
        if vitorias_t2 >= 2: return self._finalizar_mao(2)
        if self.rodada_atual > 3:
            if vitorias_t1 == 1 and vitorias_t2 == 1: return self._finalizar_mao(self.resultado_rodada[2])
            if empates == 2: return self._finalizar_mao(self.resultado_rodada[2])
            if empates == 3: return self._finalizar_mao(0)
        if empates == 1 and self.rodada_atual > 2:
            if vitorias_t1 == 1: return self._finalizar_mao(1)
            if vitorias_t2 == 1: return self._finalizar_mao(2)

    def _dar_pontos(self, time, pontos):
        if time == 1:
            self.pontos_time1 += pontos
        elif time == 2:
            self.pontos_time2 += pontos

    def resetar_estado_da_mao(self):
        self.vira = None
        self.manilhas = {}
        self.cartas_na_mesa = [] 
        self.rodada_atual = 1
        self.resultado_rodada = [0, 0, 0] 
        self.jogador_atual_idx = 0
        self.valor_mao = 1
        self.vencedor_mao = None
        self.vencedor_turno_idx = -1