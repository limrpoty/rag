class ConversationMemory:
    """Gerencia o histórico de conversas com buffer limitado"""

    def __init__(self, max_turns: int = 3):
        """
        Inicializa a memória conversacional

        Args:
            max_turns: Número máximo de turnos (pares pergunta-resposta) a manter
        """
        self.max_turns = max_turns
        self.history = []  # Lista de dicionários com 'role' e 'content'

    def add_interaction(self, user_message: str, assistant_message: str):
        """
        Adiciona uma interação completa ao histórico

        Args:
            user_message: Mensagem do usuário
            assistant_message: Resposta do assistente
        """
        self.history.append({
            'role': 'user',
            'content': user_message
        })
        self.history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        # Mantém apenas os últimos N turnos (N*2 mensagens)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_formatted_history(self) -> str:
        """
        Retorna o histórico formatado para inclusão no prompt

        Returns:
            String formatada com o histórico da conversa
        """
        if not self.history:
            return "Nenhuma conversa anterior."

        formatted = []
        for msg in self.history:
            role = "👤 Usuário" if msg['role'] == 'user' else "🤖 Assistente"
            formatted.append(f"{role}: {msg['content']}")

        return "\n\n".join(formatted)

    def clear(self):
        """Limpa todo o histórico de conversas"""
        self.history = []
        print("🧹 Memória conversacional limpa!")

    def get_turn_count(self) -> int:
        """Retorna o número de turnos (pares pergunta-resposta) no histórico"""
        return len(self.history) // 2