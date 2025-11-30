from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    """Divide texto em chunks menores mantendo contexto"""

    def __init__(self, chunk_size: int = RAGConfig.CHUNK_SIZE,
                 chunk_overlap: int = RAGConfig.CHUNK_OVERLAP):
        """
        Inicializa o chunker

        Args:
            chunk_size: Tamanho máximo de cada chunk
            chunk_overlap: Overlap entre chunks consecutivos
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Ordem de preferência para quebras
        )

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Divide texto em chunks

        Args:
            text: Texto a ser dividido
            metadata: Metadados opcionais (ex: nome do arquivo, URL)

        Returns:
            Lista de Documents do LangChain
        """
        try:
            if not text or len(text.strip()) == 0:
                raise ValueError("Texto vazio fornecido para chunking")

            # Cria chunks
            chunks = self.text_splitter.split_text(text)

            # Converte para Documents com metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata['chunk_id'] = i
                doc_metadata['chunk_total'] = len(chunks)

                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))

            return documents

        except Exception as e:
            raise Exception(f"Erro ao fazer chunking do texto: {str(e)}")