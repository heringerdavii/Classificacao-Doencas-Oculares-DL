import os
import shutil
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
# Onde seu dataset original está (a pasta que contém as pastas de classes)
BASE_PATH = '/content/gdrive/MyDrive/Colab Notebooks/Visão_trab_final/eyes/dataset' 

# Onde o dataset dividido (pastas 'train' e 'test') será criado
OUTPUT_PATH = '/content/gdrive/MyDrive/Colab Notebooks/Visão_trab_final/dataset_split' 

TEST_SIZE = 0.2  # 20% das imagens para o conjunto de teste
RANDOM_SEED = 42 # Semente fixa para garantir que a divisão seja a mesma sempre
CLASSES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'] # Suas 4 classes

# --- FUNÇÃO PARA COLETAR E COPIAR ---

def create_train_test_split(base_path, output_path, classes, test_size, random_seed):
    
    all_files = []
    all_labels = []

    # 1. Coletar caminhos de todos os arquivos
    print("Coletando caminhos dos arquivos...")
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        
        # Busca por arquivos comuns de imagem (ajuste se houver outras extensões)
        files = glob.glob(os.path.join(class_path, '*.jpeg')) + \
                glob.glob(os.path.join(class_path, '*.jpg')) + \
                glob.glob(os.path.join(class_path, '*.png'))
        
        if files:
            all_files.extend(files)
            all_labels.extend([class_name] * len(files))
        else:
            print(f"ATENÇÃO: Nenhuma imagem encontrada na pasta: {class_name}. Verifique o nome/extensões.")

    if not all_files:
        print("ERRO: Nenhuma imagem foi encontrada no dataset. Verifique o BASE_PATH.")
        return

    print(f"Total de imagens encontradas: {len(all_files)}")

    # 2. Dividir em Treino e Teste (usando a semente fixa)
    # Usando 'stratify' para manter a proporção de classes (bom para datasets desbalanceados)
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files,
        all_labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=all_labels 
    )

    print(f"Treino: {len(train_files)} imagens | Teste: {len(test_files)} imagens")

    # 3. Limpar e Criar a nova estrutura de pastas
    print("Limpando e recriando a estrutura de saída...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)


    def copy_files(file_list, label_list, split_name):
        """Copia arquivos para o novo diretório de treino ou teste."""
        count = 0
        for file_path, label in zip(file_list, label_list):
            dest_dir = os.path.join(output_path, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            count += 1
        print(f"Copiados {count} arquivos para a pasta '{split_name}'.")

    # 4. Copiar arquivos para a nova estrutura
    copy_files(train_files, train_labels, 'train')
    copy_files(test_files, test_labels, 'test')

    print("\nDivisão concluída com sucesso! A mesma divisão será usada para todos os modelos.")

# --- EXECUTAR O SCRIPT DE DIVISÃO ---
create_train_test_split(BASE_PATH, OUTPUT_PATH, CLASSES, TEST_SIZE, RANDOM_SEED)
