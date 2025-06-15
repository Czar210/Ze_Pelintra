import random
import time
import math
import pandas as pd
import copy

# Importa nossas classes de Agente e a Lógica do Jogo
from logica import JogoTruco2v2
from agente_mcts_multi import MCTSAgente as AgenteCPU
from agente_gpu import GPUAgenteMCTS as AgenteGPU

# --- CLASSE PARA REPRESENTAR NOSSOS COMPETIDORES ---
class Competidor:
    def __init__(self, nome, tipo_agente):
        self.nome = nome
        self.tipo_agente = tipo_agente
        self.agente = self._criar_agente()
        
        # Estatísticas do torneio
        self.vitorias = 0
        self.derrotas = 0
        self.pontos_feitos = 0
        self.pontos_tomados = 0

    def _criar_agente(self):
        """Instancia a classe de agente correta com base no tipo."""
        if self.tipo_agente == 'single':
            return AgenteCPU(n_simulacoes=10000, n_jobs=1)
        elif self.tipo_agente == 'multi':
            return AgenteCPU(n_simulacoes=10000, n_jobs=-1)
        elif self.tipo_agente == 'gpu':
            n_mcts_steps = 100000 // 4096 or 1
            return AgenteGPU(n_simulacoes=n_mcts_steps)
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

    while jogo.estado_jogo != "JOGO_FINALIZADO":
        if jogo.estado_jogo == "MAO_FINALIZADA":
            print(f"\r    Placar da Mão: {competidor1.nome} {jogo.pontos_time1} x {jogo.pontos_time2} {competidor2.nome}", end="", flush=True)
            time.sleep(1) # Pequena pausa para dar um efeito dramático

        estado_atual = jogo.estado_jogo
        if estado_atual in ["NOVA_MAO", "MAO_FINALIZADA"]:
            jogo.iniciar_nova_mao()
            continue
        
        # Lógica da Mão de Onze simplificada para o torneio
        elif estado_atual == "MAO_DE_ONZE":
            jogo.distribuir_cartas()
            aceitou = True # No torneio, vamos assumir que times fortes sempre jogam
            if not aceitou:
                time_em_risco = 1 if jogo.pontos_time1 >= 11 else 2
                time_adversario = 2 if time_em_risco == 1 else 1
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
    
    # Atualiza as estatísticas
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
        print(f"  {c1.nome:<20} vs  {c2.nome:<20}")
    print("="*50 + "\n")

# --- FUNÇÃO PRINCIPAL DO TORNEIO ---
def main():
    print("🏆 BEM-VINDO AO GRANDE TORNEIO DE IAs DE TRUCO! 🏆")

    # 1. Criação dos 16 Competidores
    competidores = []
    for i in range(5): competidores.append(Competidor(f"SingleCore_Bot_{i+1}", 'single'))
    for i in range(5): competidores.append(Competidor(f"MultiCore_Bot_{i+1}", 'multi'))
    for i in range(6): competidores.append(Competidor(f"GPU_Bot_{i+1}", 'gpu')) # 5 + 1 bônus

    # 2. Embaralha para criar a chave inicial
    random.shuffle(competidores)
    
    fases = {
        16: "OITAVAS DE FINAL",
        8: "QUARTAS DE FINAL",
        4: "SEMIFINAIS",
        2: "GRANDE FINAL"
    }
    
    competidores_na_rodada = competidores
    
    # 3. Loop do Torneio
    while len(competidores_na_rodada) > 1:
        nome_da_fase = fases.get(len(competidores_na_rodada), "FASE FINAL")
        print_bracket(competidores_na_rodada, nome_da_fase)
        
        vencedores_da_rodada = []
        for i in range(0, len(competidores_na_rodada), 2):
            vencedor = run_match(competidores_na_rodada[i], competidores_na_rodada[i+1])
            vencedores_da_rodada.append(vencedor)
            
        competidores_na_rodada = vencedores_da_rodada

    # 4. Anúncio do Campeão
    campeao = competidores_na_rodada[0]
    print("\n" + "👑"*50)
    print(f"O GRANDE CAMPEÃO DO TORNEIO É: {campeao.nome.upper()} !!!")
    print("👑"*50 + "\n")

    # 5. Exibição das Estatísticas Finais
    print("--- ESTATÍSTICAS FINAIS DO TORNEIO ---")
    stats_data = []
    for c in sorted(competidores, key=lambda x: x.vitorias, reverse=True):
        stats_data.append({
            "Competidor": c.nome,
            "Tipo": c.tipo_agente,
            "Vitorias": c.vitorias,
            "Derrotas": c.derrotas,
            "Pontos Feitos": c.pontos_feitos,
            "Pontos Tomados": c.pontos_tomados
        })
    
    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string())

if __name__ == '__main__':
    # Nota: A primeira vez que os agentes GPU rodam, pode haver um tempo de compilação.
    # As partidas seguintes serão mais rápidas.
    main()