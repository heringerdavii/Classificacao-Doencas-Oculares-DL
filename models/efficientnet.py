import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess 
import numpy as np
import os
import random
import pandas as pd

# Parâmetros do Projeto
NUM_CLASSES = 4 
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5 
SEEDS = [42, 10, 2023, 13, 99] 

# Função para garantir a reprodutibilidade
def set_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Variáveis para armazenar os resultados das 5 execuções
results_no_aug = []
results_with_aug = []

# 1. Definição do Modelo EfficientNetB0 
def create_efficientnet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    
    # 1. Carregar a Base EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    
    # Implementar Fine-Tuning 
    base_model.trainable = True
    fine_tune_at = 100 
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model = Sequential([
        base_model, GlobalAveragePooling2D(), Dropout(0.5), 
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def plot_confusion_matrix(conf_matrix, class_names, title):
    """Gera um mapa de calor visual da Matriz de Confusão."""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro (True Label)')
    plt.xlabel('Rótulo Previsto (Predicted Label)')
    plt.show()

#2. Preparação do Dataset e Data Augmentation 
DATASET_PATH_TRAIN = '/content/drive/MyDrive/Colab Notebooks/Visão_trab_final/dataset_split/train'
DATASET_PATH_TEST = '/content/drive/MyDrive/Colab Notebooks/Visão_trab_final/dataset_split/test'


datagen_no_aug = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess
)

datagen_with_aug = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
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


#FUNÇÃO DE AVALIAÇÃO (Com RAW Metrics)
def evaluate_model(model, generator, title, seed):
    print(f"\n--- Avaliação: {title} (Seed: {seed}) ---")

    loss, acc, prec, rec = model.evaluate(generator, steps=len(generator), verbose=0)
    Y_pred = model.predict(generator, steps=len(generator))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_score = report['weighted avg']['f1-score']
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Imprimir Matriz de Confusão em array de texto
    print("Matriz de Confusão (Teste):\n", conf_matrix) 
    # Imprimir métricas RAW para facilitar o rastreamento das 5 execuções
    print(f"RAW METRICS: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1_score:.4f}")

    CLASS_NAMES = list(generator.class_indices.keys())
    plot_confusion_matrix(conf_matrix, CLASS_NAMES, title=f"Matriz de Confusão - {title} (Seed: {seed})")
    
    return {
        'Acurácia': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1_score, 'Matriz de Confusão': conf_matrix
    }


# 3. Loop de Execução para Reprodutibilidade (5 Vezes) 
for run in range(len(SEEDS)):
    seed = SEEDS[run]
    set_seeds(seed)
    print(f"\n################ RUN {run+1} - SEED: {seed} ################")

    # Cenário 1: Sem data augmentation
    model_no_aug = create_efficientnet_model() 
    print("\n--- Treinando SEM Data Augmentation ---")
    model_no_aug.fit(
        train_generator_no_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_no_aug), verbose=1
    )
    result = evaluate_model(model_no_aug, test_generator, "EfficientNetB0 SEM Data Aug", seed)
    results_no_aug.append(result)

    # Cenário 2: Com data augmentation
    set_seeds(seed) 
    model_with_aug = create_efficientnet_model() 
    print("\n--- Treinando COM Data Augmentation ---")
    model_with_aug.fit(
        train_generator_with_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_with_aug), verbose=1
    )
    result = evaluate_model(model_with_aug, test_generator, "EfficientNetB0 COM Data Aug", seed)
    results_with_aug.append(result)

# 4. Cálculo e Apresentação Final 
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
print("\n################ RESULTADO FINAL EfficientNetB0 (SEM AUG) ################")
print(f"Acurácia (média ± dp): {stats_no_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_no_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_no_aug.get('Precision')}")
print("##########################################################################")

# Resultados para a Tabela 2 (Com Data Augmentation)
stats_with_aug = calculate_stats(results_with_aug)
print("\n################ RESULTADO FINAL EfficientNetB0 (COM AUG) ################")
print(f"Acurácia (média ± dp): {stats_with_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_with_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_with_aug.get('Precision')}")
print("##########################################################################")
