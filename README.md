# Global Solution — Classificação de Roubo e Furto com MLP

Projeto desenvolvido para a Global Solution da FIAP, utilizando redes neurais artificiais MLP (Multilayer Perceptron) para classificar ocorrências de roubo e furto de celulares na cidade de São Paulo.

O sistema utiliza dados públicos da Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP).

**Importante:** a aplicação classifica ocorrências registradas ou hipotéticas. Não prevê a ocorrência futura de crimes.

## 1. Fonte dos dados

Fonte: https://www.ssp.sp.gov.br/estatistica/consultas

Base utilizada: Celulares Subtraídos — 2025.

O projeto considera ocorrências da cidade de São Paulo, utilizando informações sobre bairro, período do dia e dia da semana.

Foram utilizados 20.000 registros no treinamento e avaliação das redes neurais.

## 2. Tecnologias utilizadas

- Python
- PyTorch
- Pandas e NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook

## 3. Metodologia

O projeto utiliza três arquiteturas MLP:

- MLP_1: uma camada oculta com 32 neurônios.
- MLP_2: duas camadas ocultas com 64 e 32 neurônios.
- MLP_3: três camadas ocultas com 128, 64 e 32 neurônios.

Os dados foram divididos em 70% para treinamento, 15% para validação e 15% para teste.

O pré-processamento utiliza One-Hot Encoding para transformar as variáveis categóricas em entradas numéricas.

A seleção da melhor arquitetura considera o F1-score de validação.

## 4. Resultados

A arquitetura selecionada foi a MLP_1.

**Resultados no conjunto de teste:**

| Métrica | Resultado |
|---|---|
| Acurácia | 71,17% |
| Precisão | 73,22% |
| Recall | 89,92% |
| F1-score | 80,71% |

Os resultados correspondem ao experimento executado com 20.000 registros.

## 5. Como executar

No terminal do VS Code, dentro da pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe modelo.py
.\.venv\Scripts\python.exe gerar_relatorio.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O treinamento utiliza a planilha oficial colocada na pasta `data/` e gera os arquivos necessários em `outputs/`.

## 6. Arquivos do projeto

- `modelo.py`: preparação dos dados, treinamento e avaliação.
- `app.py`: interface web para classificação.
- `gerar_relatorio.py`: geração do relatório PDF.
- `analise_mlp.ipynb`: notebook de análise.
- `requirements.txt`: dependências do projeto.
- `outputs/modelo.pt`: modelo treinado.
- `outputs/preprocessador.joblib`: pré-processador.
- `outputs/resultados.json`: métricas obtidas.
- `outputs/curvas.png`: curvas de treinamento.
- `outputs/matriz_confusao.png`: matriz de confusão.
- `outputs/relatorio_10_slides.pdf`: apresentação técnica.

## 7. Aplicação web

**Site publicado:**

https://gs-mlp-sao-paulo.streamlit.app/

O usuário seleciona o bairro, o período do dia e o dia da semana.

A aplicação retorna a classe estimada (roubo ou furto) e as probabilidades calculadas pelo modelo.

## 8. Repositório GitHub

https://github.com/pietrovisconti522/GS_MLP_Sao_Paulo

## 9. Limitações

O modelo utiliza padrões históricos dos registros da SSP-SP e não representa uma previsão de criminalidade.

As classificações podem apresentar erros e sofrer influência de vieses presentes nos dados originais.

As probabilidades apresentadas correspondem às estimativas do classificador, não às taxas de risco de ocorrência de crimes.

## 10. Uso de inteligência artificial generativa

Ferramentas de inteligência artificial generativa foram utilizadas como apoio à implementação, organização do código e documentação do projeto.

A equipe é responsável pela execução, revisão e interpretação dos resultados.