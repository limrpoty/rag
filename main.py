from src.rag_engine import RAGSystem

def main():
    """
    Função principal - PERSONALIZE AQUI!
    Adicione seus PDFs, sites e faça suas perguntas
    """

    print("="*70)
    print("🚀 SISTEMA RAG - 100% OPEN SOURCE (Ollama + Llama)")
    print("="*70)

    try:
        # ========================================
        # PASSO 1: Inicializa o sistema
        # ========================================
        rag = RAGSystem(model_name="llama3.2:3b")

        # ========================================
        # PASSO 2: ADICIONE SEUS PDFs AQUI ⬇️
        # ========================================
        print("\n📂 Adicionando documentos PDF...")

        # Exemplo 1: PDF único
        rag.add_document("/content/RAG-2021.pdf")
        rag.add_document("/content/plano_municipal_saude.pdf")

        # Exemplo 3: Arquivos TXT e DOCX também funcionam
        # rag.add_document("/content/arquivo.txt")
        # rag.add_document("/content/artigo.docx")

        # Exemplo 4: Lista de PDFs em loop
        # pdfs = ["/content/pdf1.pdf", "/content/pdf2.pdf", "/content/pdf3.pdf"]
        # for pdf in pdfs:
        #     rag.add_document(pdf)

        # ========================================
        # PASSO 3: ADICIONE SEUS SITES AQUI ⬇️
        # ========================================
        print("\n🌐 Adicionando sites...")

        # Exemplo 1: Site único
        rag.add_url("https://ucpel.edu.br/servicos/unidades-basicas-de-saude")

        # Exemplo 3: Lista de URLs em loop
        # urls = [
        #     "https://site1.com/artigo",
        #     "https://site2.com/noticia",
        #     "https://site3.com/pesquisa"
        # ]
        # for url in urls:
        #     rag.add_url(url)

        # ========================================
        # PASSO 4: Constrói o índice (OBRIGATÓRIO!)
        # ========================================
        rag.build_vectorstore()

        # Modo interativo com memória
        print("\n💡 Modo interativo COM MEMÓRIA INTELIGENTE ativado!")
        print("Comandos especiais:")
        print("  - 'memoria' ou 'historico': Mostra histórico")
        print("  - 'limpar': Limpa memória manualmente")
        print("  - 'auto on': Ativa limpeza automática ao mudar de assunto")
        print("  - 'auto off': Desativa limpeza automática")
        print("  - 'sair': Encerra\n")

        auto_clear = True  # Ativa limpeza automática por padrão
        print("🔄 Limpeza automática de contexto: ATIVADA\n")

        while True:
            pergunta = input("\n❓ Sua pergunta: ")

            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("👋 Encerrando...")
                break

            if pergunta.lower() in ['memoria', 'histórico', 'historico', 'memory']:
                rag.show_memory()
                continue

            if pergunta.lower() in ['limpar', 'clear', 'reset']:
                rag.clear_memory()
                continue

            if pergunta.lower() == 'auto on':
                auto_clear = True
                print("✅ Limpeza automática ATIVADA")
                continue

            if pergunta.lower() == 'auto off':
                auto_clear = False
                print("❌ Limpeza automática DESATIVADA")
                continue

            resposta = rag.query(pergunta, show_context=False, auto_clear_memory=auto_clear)
            print(f"\n📝 Resposta:\n{resposta}")

    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        print("\n🔧 Dicas:")
        print("1. Verifique se os caminhos dos arquivos estão corretos")
        print("2. Confirme que o Ollama está rodando: !ollama list")
        print("3. Teste as URLs no navegador antes de adicionar")

if __name__ == "__main__":
    main()
