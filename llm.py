import ollama

class OllamaManager:
    """Gerencia interações com o servidor Ollama"""

    @staticmethod
    def check_ollama_running() -> bool:
        """Verifica se o servidor Ollama está rodando"""
        try:
            response = ollama.list()
            return True
        except Exception:
            return False

    @staticmethod
    def check_model_available(model_name: str) -> bool:
        """Verifica se um modelo está disponível localmente"""
        try:
            models = ollama.list()
            return any(model_name in model['name'] for model in models.get('models', []))
        except Exception:
            return False

    @staticmethod
    def pull_model(model_name: str):
        """Baixa um modelo do Ollama"""
        try:
            print(f"📥 Baixando modelo {model_name}... (isso pode levar alguns minutos)")
            ollama.pull(model_name)
            print(f"✅ Modelo {model_name} baixado com sucesso!")
        except Exception as e:
            raise Exception(f"Erro ao baixar modelo: {str(e)}")

    @staticmethod
    def generate_response(model: str, prompt: str, system_prompt: str = "",
                         temperature: float = 0.7) -> str:
        """
        Gera resposta usando Ollama

        Args:
            model: Nome do modelo
            prompt: Prompt do usuário
            system_prompt: Prompt do sistema
            temperature: Temperatura para geração

        Returns:
            Resposta gerada
        """
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                options={'temperature': temperature}
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Erro na geração: {str(e)}")