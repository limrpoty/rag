import os
import pypdf
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

class DocumentLoader:
    """Carrega e processa diferentes tipos de documentos"""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Carrega arquivo de texto simples"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Tenta com outra codificação se UTF-8 falhar
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo TXT: {str(e)}")

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Carrega e extrai texto de arquivo PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo PDF: {str(e)}")

    @staticmethod
    def load_docx(file_path: str) -> str:
        """Carrega e extrai texto de arquivo DOCX"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo DOCX: {str(e)}")

    @staticmethod
    def load_file(file_path: str) -> str:
        """Detecta tipo de arquivo e carrega apropriadamente"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        extension = os.path.splitext(file_path)[1].lower()

        if extension == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif extension == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif extension == '.docx':
            return DocumentLoader.load_docx(file_path)
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {extension}")
        
class WebScraper:
    """Realiza web scraping de URLs"""

    @staticmethod
    def scrape_url(url: str) -> str:
        """
        Extrai texto de uma URL

        Args:
            url: URL para fazer scraping

        Returns:
            Texto extraído da página
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove scripts e estilos
            for script in soup(["script", "style"]):
                script.decompose()

            # Extrai texto
            text = soup.get_text()

            # Limpa o texto
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text

        except requests.exceptions.RequestException as e:
            raise Exception(f"Erro ao acessar URL {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Erro ao processar conteúdo da URL: {str(e)}")