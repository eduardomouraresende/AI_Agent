import pandas as pd
import os

class DataIngestion:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                try:
                    df = pd.read_csv(file_path)
                except:
                    df = pd.read_csv(file_path, sep=';')
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Formato nao suportado.")
            
            df.dropna(axis=1, how='all', inplace=True)
            return df
        except Exception as e:
            raise Exception(f"Falha na ingestao de dados: {str(e)}")