import unittest
import copy
from logica import JogoTruco2v2, Jogador, Carta
from agente_mc import MonteCarloBot

class TestAgenteMonteCarlo(unittest.TestCase):

    def setUp(self):
        """Prepara o ambiente para cada teste."""
        # Usamos um número baixo de simulações para o teste ser rápido
        self.bot = MonteCarloBot(n_simulacoes=10) 
        self.jogo = JogoTruco2v2(simulacao=True) # Jogo em modo silencioso
    
    def test_decisao_retorna_carta_valida(self):
        """
        Teste principal: Verifica se o bot, em um estado de jogo válido,
        retorna uma das cartas que ele realmente tem na mão.
        """
        print("\n--- Iniciando teste: test_decisao_retorna_carta_valida ---")
        
        # 1. Configurar um estado de jogo inicial
        self.jogo.distribuir_cartas()
        
        # 2. Pegar o jogador que será o nosso bot (ex: o primeiro jogador a jogar)
        jogador_bot = self.jogo.jogadores[self.jogo.jogador_atual_idx]
        mao_original_do_bot = jogador_bot.mao[:]
        
        print(f"  -> Mão do bot para o teste: {[str(c) for c in mao_original_do_bot]}")

        # 3. Chamar a função de decisão do bot
        carta_escolhida = self.bot.decidir_melhor_jogada(self.jogo, jogador_bot)

        print(f"  -> Carta escolhida pelo bot: {carta_escolhida}")

        # 4. Verificar o resultado
        self.assertIsNotNone(carta_escolhida, "O bot não escolheu nenhuma carta.")
        self.assertIsInstance(carta_escolhida, Carta, "O bot não retornou um objeto do tipo Carta.")
        self.assertIn(carta_escolhida, mao_original_do_bot, "O bot escolheu uma carta que não estava em sua mão.")
        
        print("--- Teste concluído com sucesso! ---")


if __name__ == '__main__':
    unittest.main()