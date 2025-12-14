import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50 
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
SEEDS = [42] 

def set_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

results_no_aug = []
results_with_aug = []

#  1. Definição do Modelo ResNet-50 
def create_resnet50_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    
    base_model = ResNet50(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    
    base_model.trainable = True 
    
    
    fine_tune_at = 140 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # 2. Construir a nova cabeça de classificação
    model = Sequential([
        base_model, GlobalAveragePooling2D(), Dropout(0.5), Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 3. Compilação com Learning Rate Baixo 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def plot_confusion_matrix(conf_matrix, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro (True Label)')
    plt.xlabel('Rótulo Previsto (Predicted Label)')
    plt.show()

DATASET_PATH_TRAIN = '/content/drive/MyDrive/Colab Notebooks/Dataset/train'
DATASET_PATH_TEST = '/content/drive/MyDrive/Colab Notebooks/Dataset/test'

datagen_no_aug = ImageDataGenerator(rescale=1./255)
datagen_with_aug = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
)

train_generator_no_aug = datagen_no_aug.flow_from_directory(
    DATASET_PATH_TRAIN, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)
print(f"O Generator de Treino (Sem Aug) detectou {train_generator_no_aug.num_classes} classes.")
print(f"As classes detectadas são: {train_generator_no_aug.class_indices}")

test_generator = datagen_no_aug.flow_from_directory( 
    DATASET_PATH_TEST, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
train_generator_with_aug = datagen_with_aug.flow_from_directory(
    DATASET_PATH_TRAIN, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)

def evaluate_model(model, generator, title, seed):
    print(f"\n--- Avaliação: {title} (Seed: {seed}) ---")

    loss, acc, prec, rec = model.evaluate(generator, steps=len(generator), verbose=0)
    Y_pred = model.predict(generator, steps=len(generator))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = generator.classes
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_score = report['weighted avg']['f1-score']
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Impressão da Matriz de Confusão em array de texto
    print("Matriz de Confusão (Teste):\n", conf_matrix) 
    # Impressão das métricas RAW (Acc, Prec, Rec, F1)
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

    #Cenário 1: Sem data augmentation
    model_no_aug = create_resnet50_model() 
    print("\n--- Treinando SEM Data Augmentation ---")
    model_no_aug.fit(train_generator_no_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_no_aug), verbose=1)
    result = evaluate_model(model_no_aug, test_generator, "ResNet-50 SEM Data Aug", seed)
    results_no_aug.append(result)

    #Cenário 2: Com data augmentation
    set_seeds(seed) 
    model_with_aug = create_resnet50_model() 
    print("\n--- Treinando COM Data Augmentation ---")
    model_with_aug.fit(train_generator_with_aug, epochs=EPOCHS, steps_per_epoch=len(train_generator_with_aug), verbose=1)
    result = evaluate_model(model_with_aug, test_generator, "ResNet-50 COM Data Aug", seed)
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

stats_no_aug = calculate_stats(results_no_aug)
print("\n################ RESULTADO FINAL ResNet-50 (SEM AUG) ################")
print(f"Acurácia (média ± dp): {stats_no_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_no_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_no_aug.get('Precision')}")
print("#####################################################################")

stats_with_aug = calculate_stats(results_with_aug)
print("\n################ RESULTADO FINAL ResNet-50 (COM AUG) ################")
print(f"Acurácia (média ± dp): {stats_with_aug.get('Acurácia')}")
print(f"F1-score (média ± dp): {stats_with_aug.get('F1-score')}")
print(f"Precision (média ± dp): {stats_with_aug.get('Precision')}")
print("#####################################################################")
