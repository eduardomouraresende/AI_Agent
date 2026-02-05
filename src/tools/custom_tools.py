from langchain.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from datetime import datetime
import pandas as pd
import numpy as np # <--- Importamos o NumPy aqui

@tool
def get_current_datetime(dummy: str = "") -> str:
    """
    Retorna a data e hora atual.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_pandas_tool(df: pd.DataFrame):
    """
    Cria a ferramenta de Pandas com NumPy integrado.
    """
    if df is None:
        return None

    return PythonAstREPLTool(
        # AQUI ESTÁ A MÁGICA:
        # Injetamos 'df' E 'np' no ambiente do agente.
        locals={"df": df, "np": np}, 
        
        name="python_repl_ast",
        description="""
        FERRAMENTA DE ANÁLISE DE DADOS (Pandas + NumPy).
        
        Use para executar código Python no dataframe 'df'.
        
        VARIÁVEIS DISPONÍVEIS:
        - `df`: O DataFrame pandas carregado.
        - `np`: A biblioteca numpy (já importada).
        
        REGRAS PARA CÁLCULOS:
        - Use `np` para cálculos complexos (log, exp, vetores). 
          Ex: df['log_valor'] = np.log(df['valor'])
        
        REGRAS PARA GRÁFICOS:
        1. NUNCA use plt.show().
        2. SEMPRE salve: plt.savefig('outputs/nome.png').
        3. LIMPE: plt.close().
        
        REGRAS GERAIS:
        - Use print() para ver os resultados.
        """
    )