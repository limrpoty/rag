import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import RAGConfig
from src.memory import ConversationMemory
from src.loaders import DocumentLoader, WebScraper
from src.processing import TextChunker
from src.llm import OllamaManager

class RAGSystem:
    """Sistema RAG completo com busca vetorial e geração de respostas usando Ollama"""

    def __init__(self, model_name: str = RAGConfig.OLLAMA_MODEL, memory_turns: int = 3):
        """
        Inicializa o sistema RAG com memória conversacional

        Args:
            model_name: Nome do modelo Ollama a usar
            memory_turns: Número de turnos de conversa a manter na memória
        """
        print("🔧 Inicializando Sistema RAG (100% Open Source)...")

        # Verifica se Ollama está rodando
        if not OllamaManager.check_ollama_running():
            raise Exception(
                "❌ Ollama não está rodando!\n"
                "Execute no Colab:\n"
                "!curl -fsSL https://ollama.com/install.sh | sh\n"
                "!nohup ollama serve > ollama.log 2>&1 &\n"
                "!sleep 5"
            )

        print("✅ Ollama está rodando!")

        # Verifica se modelo está disponível, senão baixa
        if not OllamaManager.check_model_available(model_name):
            print(f"⚠️  Modelo {model_name} não encontrado localmente.")
            OllamaManager.pull_model(model_name)
        else:
            print(f"✅ Modelo {model_name} disponível!")

        self.model_name = model_name

        # 🆕 CRÍTICO: Inicializa memória conversacional
        self.memory = ConversationMemory(max_turns=memory_turns)
        print(f"🧠 Memória conversacional ativada ({memory_turns} turnos)")

        # Inicializa modelo de embeddings (roda localmente, sem custo)
        print("📥 Carregando modelo de embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=RAGConfig.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Inicializa componentes
        self.chunker = TextChunker()
        self.vectorstore = None
        self.documents = []

        print("✅ Sistema RAG inicializado com sucesso!\n")

    def add_document(self, file_path: str) -> None:
        """Adiciona documento ao sistema"""
        try:
            print(f"📄 Processando arquivo: {file_path}")

            # Carrega documento
            text = DocumentLoader.load_file(file_path)

            # Cria chunks com metadata
            metadata = {
                'source': file_path,
                'source_type': 'file',
                'filename': os.path.basename(file_path)
            }
            chunks = self.chunker.chunk_text(text, metadata)

            self.documents.extend(chunks)
            print(f"✅ Arquivo processado: {len(chunks)} chunks criados\n")

        except Exception as e:
            print(f"❌ Erro ao processar arquivo: {str(e)}\n")
            raise

    def add_url(self, url: str) -> None:
        """Adiciona conteúdo de URL ao sistema"""
        try:
            print(f"🌐 Fazendo scraping da URL: {url}")

            # Faz scraping
            text = WebScraper.scrape_url(url)

            # Cria chunks com metadata
            metadata = {
                'source': url,
                'source_type': 'url'
            }
            chunks = self.chunker.chunk_text(text, metadata)

            self.documents.extend(chunks)
            print(f"✅ URL processada: {len(chunks)} chunks criados\n")

        except Exception as e:
            print(f"❌ Erro ao processar URL: {str(e)}\n")
            raise

    def build_vectorstore(self) -> None:
        """Constrói o vector store a partir dos documentos adicionados"""
        try:
            if not self.documents:
                raise ValueError("Nenhum documento foi adicionado ao sistema")

            print(f"🔨 Construindo vector store com {len(self.documents)} chunks...")

            # Cria vector store
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=RAGConfig.PERSIST_DIRECTORY
            )

            print("✅ Vector store construído com sucesso!\n")

        except Exception as e:
            print(f"❌ Erro ao construir vector store: {str(e)}\n")
            raise

    def retrieve_context(self, query: str, top_k: int = RAGConfig.TOP_K_RESULTS) -> List[Document]:
        """Recupera chunks mais relevantes para a query"""
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store não foi construído. Execute build_vectorstore() primeiro.")

            # Busca por similaridade
            results = self.vectorstore.similarity_search(query, k=top_k)

            return results

        except Exception as e:
            print(f"❌ Erro na busca: {str(e)}")
            raise

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Gera resposta usando Ollama baseado no contexto recuperado E histórico de conversa"""
        try:
            # Formata contexto dos documentos
            context = "\n\n---\n\n".join([
                f"[Fonte: {doc.metadata.get('source', 'Desconhecida')}]\n{doc.page_content}"
                for doc in context_docs
            ])

            # 🆕 Obtém histórico de conversa
            conversation_history = self.memory.get_formatted_history()

            # 🆕 NOVO: Prompt melhorado com detecção de mudança de contexto
            user_prompt = f"""=== HISTÓRICO DA CONVERSA ===
{conversation_history}

=== CONTEXTO DOS DOCUMENTOS ===
{context}

=== INSTRUÇÕES CRÍTICAS ===
1. **DETECÇÃO DE MUDANÇA DE ASSUNTO:**
   - Se a pergunta atual NÃO se relaciona com o histórico (ex: muda completamente de tema), IGNORE o histórico e responda APENAS com base nos documentos.
   - Exemplo: Se o histórico fala sobre "UBS" e a pergunta é sobre "casa de cachorro", a pergunta NÃO tem relação, então ignore o histórico.

2. **USO DO HISTÓRICO:**
   - Use o histórico APENAS quando a pergunta se refere explicitamente a algo mencionado antes (palavras como "isso", "elas", "aquilo", "o que você disse").
   - Exemplo: "Quais são os horários delas?" → "delas" se refere a algo do histórico.

3. **PRIORIDADE:**
   - SEMPRE responda com base nos DOCUMENTOS, não em inferências.
   - Se a informação NÃO está nos documentos, diga claramente: "Não encontrei essa informação nos documentos."
   - NUNCA invente informações ou repita respostas anteriores se não forem relevantes.

4. **CLAREZA:**
   - Seja direto e conciso.
   - Não repita informações já ditas a menos que seja solicitado.

=== PERGUNTA ATUAL ===
{query}

Responda de forma objetiva baseando-se APENAS nas informações dos documentos."""

            # Chama Ollama
            answer = OllamaManager.generate_response(
                model=self.model_name,
                prompt=user_prompt,
                system_prompt=RAGConfig.SYSTEM_PROMPT,
                temperature=0.3  # Baixa temperatura para respostas mais precisas
            )

            # 🆕 Adiciona interação à memória
            self.memory.add_interaction(query, answer)

            return answer

        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def clear_memory(self) -> None:
        """Limpa o histórico de conversas"""
        self.memory.clear()

    def show_memory(self) -> None:
        """Exibe o histórico atual de conversas"""
        print("\n" + "="*70)
        print("🧠 MEMÓRIA CONVERSACIONAL")
        print("="*70)
        print(f"Turnos armazenados: {self.memory.get_turn_count()}/{self.memory.max_turns}")
        print("\n" + self.memory.get_formatted_history())
        print("="*70 + "\n")

    def is_query_related_to_history(self, query: str) -> bool:
        """
        Verifica se a pergunta se relaciona com o histórico recente

        Args:
            query: Pergunta atual

        Returns:
            True se relacionada, False caso contrário
        """
        if not self.memory.history:
            return False

        # Palavras que indicam referência ao histórico
        reference_words = [
            'isso', 'aquilo', 'elas', 'eles', 'dela', 'dele', 'delas', 'deles',
            'anterior', 'antes', 'você disse', 'mencionou', 'falou', 'citou'
        ]

        query_lower = query.lower()

        # Se a pergunta contém palavras de referência, é relacionada
        if any(word in query_lower for word in reference_words):
            return True

        # Se a pergunta tem mais de 10 palavras e não tem referências, provavelmente é nova
        if len(query.split()) > 10:
            return False

        # Para perguntas curtas, assume que pode ser relacionada
        return True

    def query(self, question: str, show_context: bool = False, auto_clear_memory: bool = False) -> str:
        """
        Método principal: faz pergunta e retorna resposta

        Args:
            question: Pergunta do usuário
            show_context: Se True, mostra o contexto recuperado
            auto_clear_memory: Se True, limpa memória ao detectar mudança de assunto
        """
        try:
            print(f"\n❓ Pergunta: {question}\n")

            # 🆕 NOVO: Detecta se é uma mudança de assunto
            if auto_clear_memory and not self.is_query_related_to_history(question):
                if self.memory.get_turn_count() > 0:
                    print("🔄 Mudança de assunto detectada. Limpando memória anterior...\n")
                    self.memory.clear()

            # Recupera contexto
            print("🔍 Buscando informações relevantes...")
            context_docs = self.retrieve_context(question)

            if show_context:
                print("\n📚 Contexto recuperado:")
                for i, doc in enumerate(context_docs, 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"Fonte: {doc.metadata.get('source', 'Desconhecida')}")
                    print(f"Conteúdo: {doc.page_content[:200]}...")

            # Gera resposta
            print(f"\n💭 Gerando resposta com {self.model_name}...")
            answer = self.generate_answer(question, context_docs)

            print("\n✅ Resposta gerada!\n")
            return answer

        except Exception as e:
            error_msg = f"Erro ao processar pergunta: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            return error_msg