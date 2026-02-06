import os
import shutil
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class DatasetManager:
    def __init__(self, data_folder="data", index_folder="faiss_index"):
        self.data_folder = data_folder
        self.index_folder = index_folder
        self.vectorstore = None
        
        print("[SYSTEM] Carregando modelo de embeddings local...")
        # Modelo local leve e rápido
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Tenta carregar do disco, se falhar, cria novo
        if not self.load_index():
            self.create_index()

    def load_index(self):
        """
        Tenta carregar o banco vetorial salvo em disco.
        Retorna True se sucesso, False se não existir.
        """
        if os.path.exists(self.index_folder):
            try:
                print(f"[SYSTEM] Carregando índice vetorial existente de '{self.index_folder}'...")
                # allow_dangerous_deserialization é necessário para arquivos locais confiáveis
                self.vectorstore = FAISS.load_local(
                    self.index_folder, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("[SYSTEM] Índice carregado com sucesso! (Sem reprocessamento)")
                return True
            except Exception as e:
                print(f"[WARN] Índice corrompido ou incompatível: {e}")
                return False
        return False

    def create_index(self):
        """
        Lê arquivos, cria vetores e SALVA no disco.
        """
        print("[SYSTEM] Criando novo índice vetorial do zero...")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        docs = []
        files = [f for f in os.listdir(self.data_folder) if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if not files:
            print("[WARN] Nenhum arquivo para indexar.")
            return

        for filename in files:
            file_path = os.path.join(self.data_folder, filename)
            try:
                # Ler cabeçalho
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=5)
                else:
                    df = pd.read_excel(file_path, nrows=5)
                
                # Criar resumo semântico
                columns = ", ".join(df.columns.astype(str))
                content = f"""
                Nome do Arquivo: {filename}
                Colunas: {columns}
                Amostra: {df.head(2).to_string()}
                """
                
                docs.append(Document(
                    page_content=content, 
                    metadata={"path": file_path, "filename": filename}
                ))
            except Exception as e:
                print(f"[WARN] Erro ao ler {filename}: {e}")

        if docs:
            # Cria o banco
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # SALVA NO DISCO (A Mágica acontece aqui)
            self.vectorstore.save_local(self.index_folder)
            print(f"[SYSTEM] Índice salvo em '{self.index_folder}'. Próxima inicialização será instantânea.")
        
    def force_refresh(self):
        """
        Apaga o índice atual e recria. Usado quando o usuário adiciona novos arquivos.
        """
        if os.path.exists(self.index_folder):
            shutil.rmtree(self.index_folder)
        self.create_index()

    def find_best_dataset(self, query: str):
        if not self.vectorstore:
            return None, "Nenhum índice disponível."

        results = self.vectorstore.similarity_search(query, k=1)
        if results:
            best_doc = results[0]
            return best_doc.metadata["path"], f"Dataset encontrado: {best_doc.metadata['filename']}"
        
        return None, "Nenhum dataset relevante."