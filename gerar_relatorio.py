"""Gera 10 slides PDF a partir das métricas reais do treinamento."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'
p=OUT/'resultados.json'
if not p.exists():raise SystemExit('Treine primeiro com python modelo.py para gerar resultados reais.')
r=json.loads(p.read_text(encoding='utf-8'))
W,H=960,540;c=canvas.Canvas(str(OUT/'relatorio_10_slides.pdf'),pagesize=(W,H))
def slide(n,title,lines,images=()):
 c.setFillColor(HexColor('#101b30'));c.rect(0,0,W,H,fill=1,stroke=0)
 c.setFillColor(HexColor('#45d0ca'));c.setFont('Helvetica-Bold',15);c.drawString(48,490,'GLOBAL SOLUTION  |  MLP  |  SAO PAULO')
 c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica-Bold',26);c.drawString(48,435,title)
 y=375;c.setFont('Helvetica',16)
 for line in lines:
  c.drawString(55,y,str(line)[:105]);y-=36
 for img,x,y,w,h in images:
  if Path(img).exists():c.drawImage(str(img),x,y,w,h,preserveAspectRatio=True,anchor='c',mask='auto')
 c.setFillColor(HexColor('#93a9bf'));c.setFont('Helvetica',11);c.drawRightString(900,28,f'{n}/10');c.showPage()
slide(1,'Classificacao de Ocorrencias', ['MLP para distinguir roubo e furto na capital paulista','Integrantes: preencher nomes dos membros da equipe','Objetivo: classificar ocorrencias, nao prever crimes futuros'])
slide(2,'Problema e base de dados',[f'Fonte: SSP-SP / Celulares subtraidos | {", ".join(r["arquivos"])}',f'Amostra: {r["registros"]} registros | Furto: {r["classes"]["furto"]} | Roubo: {r["classes"]["roubo"]}','Recorte: municipio do fato = Sao Paulo; BOs duplicados removidos','Atributos: bairro, periodo do dia e dia da semana','Fonte: https://www.ssp.sp.gov.br/estatistica/consultas'])
slide(3,'Preparacao dos dados',['Natureza (rubrica) define o alvo; nao entra na MLP','Filtro de roubos/furtos exclusivos e eliminacao de BOs ambiguos','Tratamento de ausentes e derivacao de periodo/dia da semana','Codificacao one-hot aprendida somente no treino',f'Treino: {r["divisao"]["treino"]} | Validacao: {r["divisao"]["validacao"]} | Teste: {r["divisao"]["teste"]}'])
slide(4,'Arquitetura da MLP',[f'Modelo selecionado: {r["selecionada"]}',f'Camadas ocultas: {r["arquiteturas"][r["selecionada"]]["camadas"]}','Entradas: categorias codificadas por one-hot','Camadas densas + ReLU + dropout 20%','Saida: 1 logit; sigmoid converte em probabilidade de roubo'])
slide(5,'Configuracao de treinamento',['Perda: BCEWithLogitsLoss (classificacao binaria)','Otimizador: Adam | Taxa de aprendizado: 0,001','Batch size: 128 | Ate 60 epocas','Early stopping: paciencia de 8 epocas','Semente: 42 | Selecao pela metrica F1 na validacao'])
slide(6,'Tres arquiteturas avaliadas',['MLP_1: 32 neuronios (baseline simples)','MLP_2: 64 e 32 neuronios (maior capacidade)','MLP_3: 128, 64 e 32 neuronios (rede mais profunda)','Todas com ReLU e dropout 20%','Mudanca controlada: numero de camadas e neuronios'])
rows=['Modelo             Acuracia     Precision      Recall       F1']
for k,v in r['arquiteturas'].items():rows.append(f'{k:17}  {v["acuracia"]:.3f}           {v["precision"]:.3f}            {v["recall"]:.3f}       {v["f1"]:.3f}')
rows.append(f'Escolha por F1 na validacao: {r["selecionada"]}')
slide(7,'Comparacao na validacao',rows)
slide(8,'Curvas e matriz de confusao',[f'Teste final: F1={r["teste"]["f1"]:.3f} | Acuracia={r["teste"]["acuracia"]:.3f}','Diagnostico: comparar distancias entre curvas de treino e validacao'],[(OUT/'curvas.png',45,65,460,260),(OUT/'matriz_confusao.png',515,65,390,260)])
slide(9,'Aplicacao web',['Interface Streamlit: bairro, periodo e dia da semana','Botao para classificar ocorrencia e mostrar probabilidades','Preprocessador e pesos carregados do treinamento','Executar: streamlit run app.py','Inserir captura real da interface apos executar a aplicacao'])
slide(10,'Conclusoes e limitacoes',[f'Melhor modelo: {r["selecionada"]} | F1 teste: {r["teste"]["f1"]:.3f}','O classificador reconhece padroes de registros; nao estima risco','Limitacoes: vies de notificacao, cobertura e dados historicos','IA generativa: apoio em codigo, documentacao e organizacao','Validacao e interpretacao dos resultados cabem a equipe'])
c.save();print(OUT/'relatorio_10_slides.pdf')
