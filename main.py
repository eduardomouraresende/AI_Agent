import sys
import os
import matplotlib.pyplot as plt

# Imports explícitos
from src.data_ingestion import DataIngestion
from src.agent_core import DataAnalysisAgent
from src.file_manager import FileManager

def main():
    print("🤖 --- Agente de Dados (Modo Chat) ---")
    print(f"[Info] Pasta de gráficos: ./outputs")
    print(f"[Info] Digite 'sair' para encerrar ou 'trocar arquivo' para mudar a base.")
    
    file_manager = FileManager()
    
    # Variável que mantém o agente vivo entre as mensagens
    active_analyzer = None 
    current_file_path = None

    while True:
        # Prompt limpo, sem [sem arquivo]
        try:
            user_input = input("\nVocê: ").strip()
        except KeyboardInterrupt:
            break
            
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break

        if not user_input:
            continue

        # --- CENÁRIO 1: Nenhum agente ativo (Primeiro acesso ou troca) ---
        if not active_analyzer:
            # 1. Tenta identificar qual arquivo o usuário quer
            print("⏳ Procurando base de dados correspondente...")
            file_path, error = file_manager.select_file_based_on_query(user_input)
            
            if file_path:
                filename = os.path.basename(file_path)
                
                # 2. CONFIRMAÇÃO DE ARQUIVO (Requisito Solicitado)
                confirm = input(f"🤖 Encontrei o arquivo '{filename}'. Deseja utilizá-lo para esta análise? (s/n): ").strip().lower()
                
                if confirm in ['s', 'sim', 'y', 'yes']:
                    try:
                        print(f"📂 Carregando {filename}...")
                        df = DataIngestion.load_data(file_path)
                        
                        # Inicializa o agente
                        active_analyzer = DataAnalysisAgent(df)
                        current_file_path = file_path
                        
                        print("✅ Base carregada. Iniciando análise...")
                        
                        # 3. Envia a pergunta original para o agente começar o planejamento
                        response_text = active_analyzer.process_message(user_input)
                        print(f"\n🤖 Agente: {response_text}")
                        
                    except Exception as e:
                        print(f"❌ Erro ao carregar: {e}")
                else:
                    print("🤖 Ok, busca cancelada. Por favor, tente reformular ou verifique a pasta 'data'.")
            else:
                print(f"🤖 {error}")

        # --- CENÁRIO 2: Conversa Contínua (Agente já sabe o arquivo) ---
        else:
            # Comando para resetar
            if "trocar arquivo" in user_input.lower() or "nova base" in user_input.lower():
                active_analyzer = None
                current_file_path = None
                print("🔄 Memória limpa. Qual a próxima base de dados?")
                continue

            # Processamento normal da conversa
            print("⏳ Pensando...")
            response_text = active_analyzer.process_message(user_input)
            print(f"\n🤖 Agente: {response_text}")

if __name__ == "__main__":
    main()