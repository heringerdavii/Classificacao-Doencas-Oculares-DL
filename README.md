 Classificação de Doenças Oculares com Deep Learning

Este repositório contém os scripts para o trabalho prático de classificação de 4 doenças oculares usando as arquiteturas **CNN básica, ResNet-50 e Efficient Net**.

# Resumo do Projeto

Projeto de classificação de 4 doenças oculares via Deep Learning. Implementação e comparação de 4 arquiteturas (CNN, VGG19, ResNet-50, Efficient Net). O projeto avalia o impacto do Data Augmentation e a estabilidade dos modelos através de 5 execuções, reportando Acurácia, F1-score e Matriz de Confusão.

## 1. Configuração e Dependências

### 1.1 Instalação

O ambiente recomendado é o **Google Colab** com aceleração **GPU**. Instale as dependências usando o requirements.txt:

bash
!pip install -r requirements.txt

1.2 Montar o Google Drive
O dataset dividido (dataset_split) deve estar no Google Drive. 
Execute o código abaixo no Colab:

from google.colab import drive
drive.mount('/content/gdrive')

2. Preparação do Dataset (Divisão Train/Test)
É OBRIGATÓRIO rodar este script APENAS UMA VEZ para criar a divisão consistente (Train/Test) utilizada por todos os 4 modelos.

2.1 Script de Divisão (data_split/dataset_splitter.py)
Acesse o arquivo data_split/dataset_splitter.py e AJUSTE OS CAMINHOS BASE_PATH E OUTPUT_PATH antes de executar.

2.2 Caminhos Usados no Projeto
Todos os scripts de treinamento (/models/*.py) utilizam os seguintes caminhos, gerados pelo script de divisão:

DATASET_PATH_TRAIN '/content/gdrive/MyDrive/Caminho/dataset_split/train'
DATASET_PATH_TEST = '/content/gdrive/MyDrive/Caminho/dataset_split/test'


3. Execução dos Modelos e Reprodutibilidade
   
3.1 Parâmetros Chave (Obrigatórios)
Todos os scripts (/models/*.py) estão configurados com:
Épocas: 50
Sementes: 5 sementes diferentes (SEEDS = [42, 10, 2023, 13, 99]) para garantir a estabilidade dos resultados.
Matriz de Confusão: Plotada visualmente em mapa de calor a cada avaliação.

3.2 Executando a CNN Básica
Execute o código no arquivo /models/cnn_basic.py. O script treinará e avaliará os cenários Sem e Com Data Augmentation por 5 vezes, imprimindo a Média e o Desvio Padrão final.
