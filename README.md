# Global Solution — Classificação de roubo e furto com MLP

Projeto de classificação binária de ocorrências na **cidade de São Paulo**, baseado em dados públicos da SSP-SP. Não prevê se um crime ocorrerá.

## Fonte e recorte

Fonte: https://www.ssp.sp.gov.br/estatistica/consultas — seção **Bases de Dados > Celulares subtraídos**. Baixe um arquivo CSV ou XLSX de **2025** e coloque-o em `data/`. O código filtra **cidade do fato = São Paulo**, não a delegacia do registro; mantém apenas rubricas exclusivamente de roubo ou furto; elimina boletins de classificação ambígua e registros repetidos por chave de BO. O recorte temporal efetivo deve ser confirmado no arquivo baixado.

Atributos: bairro do fato, período do dia (derivado da hora do fato) e dia da semana (derivado da data do fato). O rótulo vem de RUBRICA e **não é entrada da rede**. Sem nomes, endereços completos ou identificadores pessoais no modelo.

## Execução (Windows / VS Code)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python modelo.py
streamlit run app.py
```

O treinamento gera `outputs/modelo.pt`, `outputs/preprocessador.joblib`, `outputs/resultados.json`, `outputs/curvas.png` e `outputs/matriz_confusao.png`. Execute `python gerar_relatorio.py` para produzir `outputs/relatorio_10_slides.pdf` **após o treinamento**.

## Notebook

Abra `analise_mlp.ipynb` no VS Code ou Jupyter, com o ambiente virtual selecionado, e execute todas as células na ordem.

## Metodologia

- Divisão estratificada 70% treino, 15% validação e 15% teste, com semente 42.
- One-hot encoding ajustado apenas no treino; desconhecidos são ignorados.
- Três redes densas com ReLU, dropout 0,2 e saída logit binária.
- BCEWithLogitsLoss, Adam (0,001), batch 128, até 60 épocas e early stopping (paciência 8).
- Seleção por F1 na validação, desempate por acurácia; teste usado uma única vez ao final.
- Acurácia, precision, recall e F1 usam **roubo** como classe positiva.

## Aplicação online

**URL pública:** PENDENTE DE PUBLICAÇÃO. Para atender à entrega, publique no Streamlit Community Cloud conectando este repositório e definindo `app.py` como arquivo principal. Inclua os artefatos `modelo.pt` e `preprocessador.joblib` gerados a partir da base pública no repositório, ou configure um processo de geração no deploy. O `.gitignore` bloqueia os artefatos por padrão; para publicar, remova as duas regras de `outputs/` ou use `git add -f` para esses dois arquivos.

## Limitações

A classificação usa apenas padrões de registros históricos e não substitui análise policial. Pode reproduzir vieses de notificação e variar por período. A deduplicação por BO evita contar vários itens do mesmo boletim como exemplos separados. Sem base oficial baixada, **não existem métricas legítimas nem modelo treinado**.

## Uso de IA generativa

IA generativa apoiou a implementação, a documentação e a organização do projeto. A equipe é responsável por executar, revisar e explicar o código e os resultados.
