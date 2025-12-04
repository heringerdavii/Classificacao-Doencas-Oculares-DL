import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

# Função para garantir a reprodutibilidade (fixar todas as sementes)
def set_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Variáveis para armazenar os resultados das 5 execuções
results_no_aug = []
results_with_aug = []

# 1. Definição do Modelo CNN Básica
def create_basic_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


# FUNÇÃO PARA PLOTAR MATRIZ DE CONFUSÃO
def plot_confusion_matrix(conf_matrix, class_names, title):
    """Gera um mapa de calor visual da Matriz de Confusão."""
    
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


# 2. Preparação do Dataset e Data Augmentation 

DATASET_PATH_TRAIN = '/caminho/dataset_split/train'
DATASET_PATH_TEST = '/caminho/dataset_split/train''


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
    DATASET_PATH_TRAIN,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
print(f"O Generator de Treino (Sem Aug) detectou {train_generator_no_aug.num_classes} classes.")
print(f"As classes detectadas são: {train_generator_no_aug.class_indices}")

# Carregador de Teste
test_generator = datagen_no_aug.flow_from_directory( 
    DATASET_PATH_TEST,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Carregador de Treino (Com Augmentation)
train_generator_with_aug = datagen_with_aug.flow_from_directory(
    DATASET_PATH_TRAIN,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# --- FUNÇÃO DE AVALIAÇÃO ---
def evaluate_model(model, generator, title, seed):
    print(f"\n--- Avaliação: {title} (Seed: {seed}) ---")

    # 1. Avaliação do Keras (Loss, Accuracy, Precision, Recall)
    loss, acc, prec, rec = model.evaluate(generator, steps=len(generator), verbose=0)

    # 2. Previsões para F1-score e Matriz de Confusão
    Y_pred = model.predict(generator, steps=len(generator))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes
    
    # Classificação do F1-score
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_score = report['weighted avg']['f1-score']

    # Matriz de Confusão (Cálculo)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # NOVO: PLOTAR A MATRIZ VISUALMENTE
    CLASS_NAMES = list(generator.class_indices.keys())
    plot_confusion_matrix(
        conf_matrix, 
        CLASS_NAMES, 
        title=f"Matriz de Confusão - {title} (Seed: {seed})"
    )
    
 
    # Coletar resultados
    return {
        'Acurácia': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1_score,
        'Matriz de Confusão': conf_matrix 
    }



    # 3. Loop de Execução para Reprodutibilidade (5 Vezes) 
# O loop vai rodar 5 vezes, uma para cada semente em SEEDS

for run in range(len(SEEDS)):
    seed = SEEDS[run]
    set_seeds(seed)
    print(f"\n################ RUN {run+1} - SEED: {seed} ################")



    # CENÁRIO 1: SEM DATA AUGMENTATION
    cnn_model_no_aug = create_basic_cnn()
    print("\n--- Treinando SEM Data Augmentation ---")
    cnn_model_no_aug.fit(
        train_generator_no_aug,
        epochs=EPOCHS,
        steps_per_epoch=len(train_generator_no_aug),
        verbose=1
    )

    result = evaluate_model(cnn_model_no_aug, test_generator, "CNN SEM Data Aug", seed)
    results_no_aug.append(result)



    # CENÁRIO 2: COM DATA AUGMENTATION 
    set_seeds(seed) # Resetar semente antes de re-criar/treinar novo modelo
    cnn_model_with_aug = create_basic_cnn()
    print("\n--- Treinando COM Data Augmentation ---")
    cnn_model_with_aug.fit(
        train_generator_with_aug,
        epochs=EPOCHS,
        steps_per_epoch=len(train_generator_with_aug),
        verbose=1
    )

    result = evaluate_model(cnn_model_with_aug, test_generator, "CNN COM Data Aug", seed)
    results_with_aug.append(result)



# 4. Cálculo e Apresentação Final (CNN Básica) 
def calculate_stats(results):
    df = pd.DataFrame(results).drop(columns=['Matriz de Confusão'])
    mean = df.mean().to_dict()
    std = df.std().to_dict()

    final_results = {}

    for key in mean:
        final_results[key] = f"{mean[key]:.4f} ± {std[key]:.4f}"   # Formato: Média ± DP
    return final_results

# Resultados para a Tabela 1 (Sem Data Aug)
stats_no_aug = calculate_stats(results_no_aug)
print("\n################ RESULTADO FINAL CNN (SEM AUG) ################")
print(f"Acurácia (média ± dp): {stats_no_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_no_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_no_aug.get('Precision')}")
print("###############################################################")

# Resultados para a Tabela 2 (Com Data Aug)
stats_with_aug = calculate_stats(results_with_aug)
print("\n################ RESULTADO FINAL CNN (COM AUG) ################")
print(f"Acurácia (média ± dp): {stats_with_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_with_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_with_aug.get('Precision')}")
print("###############################################################")
