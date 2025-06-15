import sys
from logica import JogoTruco2v2
from agente_gpu import GPUAgenteMCTS
import random

def test_gpu_rollout():
    """
    Função de teste para executar o _gpu_rollout e verificar se ele
    roda sem erros e produz um resultado sensato.
    """
    print("="*40)
    print("--- INICIANDO TESTE DO KERNEL GPU ---")
    print("="*40)

    try:
        # 1. Configuração
        print("\n[1] Configurando o ambiente de teste...")
        # Instancia o agente da GPU.
        agente = GPUAgenteMCTS()
        agente.n_rollouts_por_decisao = 4096 # Um bom número para um teste rápido

        # Cria um estado de jogo inicial aleatório
        jogo = JogoTruco2v2()
        jogo.iniciar_nova_mao()
        
        bot_id = 1
        jogador_bot = jogo.jogadores[bot_id - 1]

        print("\n--- Estado Inicial do Jogo (Visão da CPU) ---")
        print(f"Bot (Jogador {bot_id}) Mão: {[str(c) for c in jogador_bot.mao]}")
        print(f"Vira: {jogo.vira}")
        print(f"Jogador que começa: {jogo.jogadores[jogo.jogador_atual_idx].id}")
        print("--------------------------------------------\n")

        # 2. Execução
        print("[2] Executando o método _gpu_rollout...")
        
        # Chamamos diretamente o método que queremos testar
        taxa_de_vitoria = agente._gpu_rollout(jogo, bot_id)

        # 3. Verificação
        print("\n[3] Verificando o resultado...")
        if 0.0 <= taxa_de_vitoria <= 1.0:
            print(f"SUCESSO! O kernel retornou uma taxa de vitória válida: {taxa_de_vitoria:.2%}")
        else:
            print(f"FALHA! O resultado '{taxa_de_vitoria}' não é uma probabilidade válida.")

    except Exception as e:
        print("\n" + "!"*40)
        print(" OCORREU UM ERRO DURANTE O TESTE DA GPU")
        print(f" Erro: {e}")
        print("!"*40)
        # Imprime o traceback completo para ajudar na depuração
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "="*40)
        print("--- TESTE CONCLUÍDO ---")
        print("="*40)

if __name__ == "__main__":
    test_gpu_rollout()