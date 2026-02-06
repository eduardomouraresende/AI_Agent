from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from datetime import datetime
import pandas as pd
import numpy as np
from src.data_ingestion import DataIngestion

# Variável global para armazenar o contexto compartilhado
# Isso permite que a ferramenta 'load_dataset' altere o 'df' 
# e a ferramenta 'python_repl' veja a alteração.
shared_context = {
    "df": None, 
    "np": np
}

@tool
def get_current_datetime(dummy: str = "") -> str:
    """Retorna data e hora atual."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_analysis_tools(dataset_manager):
    """
    Retorna a lista de ferramentas configuradas, incluindo a capacidade de carregar dados.
    Recebe o dataset_manager inicializado.
    """
    
    # Tool 1: Carregar Dataset via Busca Semântica
    @tool
    def load_relevant_dataset(query: str) -> str:
        """
        Use esta ferramenta quando o usuário perguntar sobre um assunto novo 
        e você precisar encontrar o arquivo CSV/Excel correto.
        Input: Uma frase descrevendo o que você busca (ex: 'dados de vendas 2024').
        Output: O resultado do carregamento.
        """
        file_path, msg = dataset_manager.find_best_dataset(query)
        
        if file_path:
            try:
                # Carrega o novo DF
                new_df = DataIngestion.load_data(file_path)
                
                # ATUALIZA O CONTEXTO COMPARTILHADO
                shared_context["df"] = new_df
                
                # Retorna resumo para o agente saber o que tem na mão
                columns = ", ".join(new_df.columns.astype(str))
                return f"SUCESSO: {msg}. Colunas carregadas: {columns}. Agora use a tool 'python_repl_ast' para analisar."
            except Exception as e:
                return f"Erro ao ler arquivo: {e}"
        else:
            return "Não encontrei nenhum arquivo relevante para esse tema."

    # Tool 2: Python REPL (Conectado ao shared_context)
    # Note que passamos o dicionário shared_context inteiro, não o valor df.
    python_tool = PythonAstREPLTool(
        locals=shared_context, 
        name="python_repl_ast",
        description="""
        FERRAMENTA DE ANÁLISE (Pandas/Numpy).
        Use para analisar os dados carregados na variável `df`.
        Se `df` for None ou vazio, use a ferramenta `load_relevant_dataset` primeiro.
        Sempre use print().
        Sempre salve gráficos em 'outputs/'.
        """
    )

    return [get_current_datetime, load_relevant_dataset, python_tool]