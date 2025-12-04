import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50 # NOVO IMPORT para ResNet50
import numpy as np
import os
import random
import pandas as pd

# Parâmetros do Projeto
NUM_CLASSES = 4 
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50 
SEEDS = [42, 10, 2023, 13, 99] 

def set_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Variáveis para armazenar os resultados das 5 execuções
results_no_aug = []
results_with_aug = []

# 1. Definição do Modelo ResNet-50 (Transfer Learning) 
def create_resnet50_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    
    # 1. Carregar a Base ResNet-50 (Congelar os pesos)
    base_model = ResNet50(
        weights='imagenet',          # Usa pesos pré-treinados
        include_top=False,           # Não inclui as camadas finais de classificação
        input_shape=input_shape      # Usa sua resolução de entrada
    )
    
    # Congelar as camadas da base
    base_model.trainable = False 
    
    # 2. Construir a nova cabeça de classificação (Head)
    model = Sequential([
        base_model, # Inclui a base ResNet-50
        
        GlobalAveragePooling2D(), # Reduz a dimensionalidade média do tensor de características
        Dropout(0.5),            # Regularização
        Dense(256, activation='relu'),
        
        # Camada de Saída (4 classes)
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilação do modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# 🎨FUNÇÃO PARA PLOTAR MATRIZ DE CONFUSÃO 
def plot_confusion_matrix(conf_matrix, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d',    
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro (True Label)')
    plt.xlabel('Rótulo Previsto (Predicted Label)')
    plt.show()

#  2. Preparação do Dataset e Data Augmentation 
DATASET_PATH_TRAIN = '/content/drive/MyDrive/Colab Notebooks/Visão_trab_final/dataset_split/train'
DATASET_PATH_TEST = '/content/drive/MyDrive/Colab Notebooks/Visão_trab_final/dataset_split/test'

# Restante da Seção 2 e Criação dos Generators
datagen_no_aug = ImageDataGenerator(rescale=1./255)
datagen_with_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Carregador de Treino (Sem Augmentation)
train_generator_no_aug = datagen_no_aug.flow_from_directory(
    DATASET_PATH_TRAIN, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)
print(f"O Generator de Treino (Sem Aug) detectou {train_generator_no_aug.num_classes} classes.")
print(f"As classes detectadas são: {train_generator_no_aug.class_indices}")

# Carregador de Teste
test_generator = datagen_no_aug.flow_from_directory( 
    DATASET_PATH_TEST, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Carregador de Treino (Com Augmentation)
train_generator_with_aug = datagen_with_aug.flow_from_directory(
    DATASET_PATH_TRAIN, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)


# --- FUNÇÃO DE AVALIAÇÃO (Sem Alteração) ---
def evaluate_model(model, generator, title, seed):
    print(f"\n--- Avaliação: {title} (Seed: {seed}) ---")

    loss, acc, prec, rec = model.evaluate(generator, steps=len(generator), verbose=0)
    Y_pred = model.predict(generator, steps=len(generator))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_score = report['weighted avg']['f1-score']
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    CLASS_NAMES = list(generator.class_indices.keys())
    plot_confusion_matrix(conf_matrix, CLASS_NAMES, title=f"Matriz de Confusão - {title} (Seed: {seed})")
    
    # NOVO: Imprimir métricas RAW para facilitar o rastreamento das 5 execuções
    print(f"RAW METRICS: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1_score:.4f}")

    return {
        'Acurácia': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1_score, 'Matriz de Confusão': conf_matrix
    }


# 3. Loop de Execução para Reprodutibilidade (5 Vezes)
# AGORA O LOOP USA AS 50 ÉPOCAS E AS 5 SEMENTES
for run in range(len(SEEDS)):
    seed = SEEDS[run]
    set_seeds(seed)
    print(f"\n################ RUN {run+1} - SEED: {seed} ################")

    # --- CENÁRIO 1: SEM DATA AUGMENTATION ---
    cnn_model_no_aug = create_resnet50_model() # USANDO RESNET-50
    print("\n--- Treinando SEM Data Augmentation ---")
    cnn_model_no_aug.fit(
        train_generator_no_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_no_aug), verbose=1
    )
    result = evaluate_model(cnn_model_no_aug, test_generator, "ResNet-50 SEM Data Aug", seed)
    results_no_aug.append(result)

    # --- CENÁRIO 2: COM DATA AUGMENTATION ---
    set_seeds(seed) 
    cnn_model_with_aug = create_resnet50_model() # USANDO RESNET-50
    print("\n--- Treinando COM Data Augmentation ---")
    cnn_model_with_aug.fit(
        train_generator_with_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_with_aug), verbose=1
    )
    result = evaluate_model(cnn_model_with_aug, test_generator, "ResNet-50 COM Data Aug", seed)
    results_with_aug.append(result)

#4. Cálculo e Apresentação Final (ResNet-50)
def calculate_stats(results):
    df = pd.DataFrame(results).drop(columns=['Matriz de Confusão'])
    mean = df.mean().to_dict()
    std = df.std().to_dict()

    final_results = {}
    for key in mean:
        final_results[key] = f"{mean[key]:.4f} ± {std[key]:.4f}"
    return final_results

# Resultados para a Tabela 1 (Sem Data Augmentation)
stats_no_aug = calculate_stats(results_no_aug)
print("\n################ RESULTADO FINAL ResNet-50 (SEM AUG) ################")
print(f"Acurácia (média ± dp): {stats_no_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_no_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_no_aug.get('Precision')}")
print("#####################################################################")

# Resultados para a Tabela 2 (Com Data Augmentation)
stats_with_aug = calculate_stats(results_with_aug)
print("\n################ RESULTADO FINAL ResNet-50 (COM AUG) ################")
print(f"Acurácia (média ± dp): {stats_with_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_with_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_with_aug.get('Precision')}")
print("#####################################################################")
