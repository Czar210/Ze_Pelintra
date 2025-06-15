# tournament.py
import random
import time
import pandas as pd
import copy

# Importa a lógica do jogo e as nossas classes de Agente dos arquivos corretos
from logica import JogoTruco2v2
from time_limit_mcts import MCTSAgente as AgenteCPU
from time_limit_gpu import GPUAgenteMCTS as AgenteGPU

# --- CLASSE PARA REPRESENTAR NOSSOS COMPETIDORES ---
class Competidor:
    def __init__(self, nome, tipo_agente, time_limit=30):
        self.nome = nome
        self.tipo_agente = tipo_agente
        self.time_limit = time_limit
        self.agente = self._criar_agente()
        
        # Estatísticas do torneio
        self.vitorias = 0
        self.derrotas = 0
        self.pontos_feitos = 0
        self.pontos_tomados = 0

    def _criar_agente(self):
        """Instancia a classe de agente correta com base no tipo."""
        print(f"Criando competidor: {self.nome} (Tipo: {self.tipo_agente})...")
        if self.tipo_agente == 'single':
            return AgenteCPU(time_limit_por_jogada=self.time_limit, n_jobs=1)
        elif self.tipo_agente == 'multi':
            return AgenteCPU(time_limit_por_jogada=self.time_limit, n_jobs=-1)
        elif self.tipo_agente == 'gpu':
            return AgenteGPU(time_limit_por_jogada=self.time_limit)
        return None

    def __repr__(self):
        return self.nome

# --- FUNÇÃO PARA EXECUTAR UMA PARTIDA ---
def run_match(competidor1, competidor2):
    """
    Executa uma partida de Truco entre dois competidores, com interface visual.
    """
    print(f"  > Iniciando partida: {competidor1.nome} (Time 1) vs {competidor2.nome} (Time 2)")
    
    jogo = JogoTruco2v2(simulacao=True) # Silencioso na lógica, nós controlamos o print
    bot_t1 = competidor1.agente
    bot_t2 = competidor2.agente

    # O loop do jogo foi copiado para cá
    while jogo.estado_jogo != "JOGO_FINALIZADO":
        if jogo.estado_jogo == "MAO_FINALIZADA":
            print(f"\r    Placar da Mão: {competidor1.nome} {jogo.pontos_time1} x {jogo.pontos_time2} {competidor2.nome}   ", end="", flush=True)
            time.sleep(0.5)

        estado_atual = jogo.estado_jogo
        if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]:
            jogo.iniciar_nova_mao()
            continue
        
        elif estado_atual == "MAO_DE_ONZE":
            jogo.distribuir_cartas()
            time_em_risco_id = 1 if jogo.pontos_time1 >= 11 else 2
            bot_em_risco = bot_t1 if time_em_risco_id == 1 else bot_t2
            jogador_em_risco = jogo.jogadores[0] if time_em_risco_id == 1 else jogo.jogadores[1]
            aceitou = bot_em_risco.decidir_mao_de_onze_com_mc(jogo, jogador_em_risco)
            
            if not aceitou:
                time_adversario = 2 if time_em_risco_id == 1 else 1
                jogo._dar_pontos(time_adversario, 1)
                jogo.estado_jogo = "MAO_FINALIZADA"
            continue

        elif estado_atual == "EM_ANDAMENTO":
            jogador_da_vez = jogo.jogadores[jogo.jogador_atual_idx]
            bot_da_vez = bot_t1 if jogador_da_vez.time_id == 1 else bot_t2
            
            if not jogador_da_vez.mao:
                jogo._checar_vencedor_da_mao()
                if jogo.estado_jogo == "JOGO_FINALIZADO": break
                continue

            carta_jogada, _ = bot_da_vez.decidir_melhor_jogada(jogo, jogador_da_vez)
            if carta_jogada:
                jogo.jogar_carta(jogador_da_vez.id, carta_jogada)
            if jogo.estado_jogo == "JOGO_FINALIZADO":
                break
    
    # Atualiza e exibe as estatísticas
    vencedor = None
    if jogo.pontos_time1 >= 12:
        vencedor = competidor1
        competidor1.vitorias += 1
        competidor2.derrotas += 1
    else:
        vencedor = competidor2
        competidor2.vitorias += 1
        competidor1.derrotas += 1

    competidor1.pontos_feitos += jogo.pontos_time1
    competidor1.pontos_tomados += jogo.pontos_time2
    competidor2.pontos_feitos += jogo.pontos_time2
    competidor2.pontos_tomados += jogo.pontos_time1
    
    print(f"\r    FIM DE JOGO! Placar final: {competidor1.nome} {jogo.pontos_time1} x {jogo.pontos_time2} {competidor2.nome}")
    print(f"    Vencedor: {vencedor.nome}\n")
    return vencedor

# --- FUNÇÃO PARA IMPRIMIR A CHAVE DO TORNEIO ---
def print_bracket(competidores, nome_fase):
    print("\n" + "="*50)
    print(f"| {nome_fase.center(48)} |")
    print("="*50)
    for i in range(0, len(competidores), 2):
        c1 = competidores[i]
        c2 = competidores[i+1]
        print(f"  {c1.nome:<22} vs  {c2.nome:<22}")
    print("="*50 + "\n")
    time.sleep(2) # Pausa dramática

# --- FUNÇÃO PRINCIPAL DO TORNEIO ---
def main():
    print("🏆 BEM-VINDO AO GRANDE TORNEIO DE IAs DE TRUCO! 🏆\n")

    competidores = []
    for i in range(2): competidores.append(Competidor(f"SingleCore_Bot_{i+1}", 'single'))
    for i in range(3): competidores.append(Competidor(f"MultiCore_Bot_{i+1}", 'multi'))
    for i in range(3): competidores.append(Competidor(f"GPU_Bot_{i+1}", 'gpu'))

    random.shuffle(competidores)
    
    fases = {8: 'QUARTAS DE FINAL', 4: "SEMIFINAIS", 2: "GRANDE FINAL"}
    competidores_na_rodada = competidores
    
    while len(competidores_na_rodada) > 1:
        nome_da_fase = fases.get(len(competidores_na_rodada), "FASE FINAL")
        print_bracket(competidores_na_rodada, nome_da_fase)
        
        vencedores_da_rodada = []
        for i in range(0, len(competidores_na_rodada), 2):
            vencedor = run_match(competidores_na_rodada[i], competidores_na_rodada[i+1])
            vencedores_da_rodada.append(vencedor)
            
        competidores_na_rodada = vencedores_da_rodada

    campeao = competidores_na_rodada[0]
    print("\n" + "👑"*50)
    print(f"O GRANDE CAMPEÃO DO TORNEIO É: {campeao.nome.upper()} !!!")
    print("👑"*50 + "\n")

    print("--- ESTATÍSTICAS FINAIS DO TORNEIO ---")
    stats_data = []
    for c in sorted(competidores, key=lambda x: (x.vitorias, x.pontos_feitos - x.pontos_tomados), reverse=True):
        stats_data.append({
            "Competidor": c.nome,
            "Tipo": c.tipo_agente,
            "Vitorias": c.vitorias,
            "Derrotas": c.derrotas,
            "Pontos Feitos": c.pontos_feitos,
            "Pontos Tomados": c.pontos_tomados,
            "Saldo": c.pontos_feitos - c.pontos_tomados
        })
    
    df_stats = pd.DataFrame(stats_data)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_stats)

if __name__ == '__main__':
    main()