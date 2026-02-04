import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class FileManager:
    """
    Gerencia e seleciona arquivos automaticamente baseado no contexto.
    """
    def __init__(self, directory="data"):
        # Ajusta o caminho para ser relativo à execução do main.py
        self.directory = os.path.join(os.getcwd(), directory)
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def list_files(self):
        """Lista arquivos não ocultos no diretório de dados."""
        return [f for f in os.listdir(self.directory) if not f.startswith('.')]

    def select_file_based_on_query(self, user_query: str):
        """
        Usa o LLM para decidir qual arquivo carregar.
        """
        files = self.list_files()
        
        if not files:
            return None, "O diretorio de dados esta vazio."

        template = """
        System: Voce e um roteador de arquivos.
        Arquivos disponiveis: {file_list}
        Input do usuario: "{user_query}"
        
        Instrucao: Retorne APENAS o nome exato do arquivo que melhor atende ao pedido.
        Se nenhum for relevante, retorne "None".
        """
        
        prompt = PromptTemplate(
            input_variables=["file_list", "user_query"],
            template=template
        )
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "file_list": ", ".join(files),
                "user_query": user_query
            })
            
            filename = response.content.strip()
            
            if filename == "None" or filename not in files:
                return None, f"Nenhum arquivo relevante encontrado para a query."
                
            return os.path.join(self.directory, filename), None
            
        except Exception as e:
            return None, f"Erro na selecao de arquivo: {str(e)}"