"""Treinamento reproduzível da classificação de ocorrências SSP-SP."""
from pathlib import Path
import json, re, unicodedata, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
FEATURES=['bairro','periodo','dia_semana']
CONFIGS={'MLP_1':[32], 'MLP_2':[64,32], 'MLP_3':[128,64,32]}

def norm(s):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().upper()
    return re.sub(r'[^A-Z0-9]+','_',s).strip('_')

def read_file(p):
    if p.suffix.lower() in ('.xlsx','.xls'):
        return pd.read_excel(p, sheet_name='CELULAR_2025' if p.name.lower() == 'celularessubtraidos_2025.xlsx' else 0)
    for enc in ('utf-8-sig','latin1'):
        for sep in ('; ', ';', ','):
            if sep=='; ':continue
            try:
                df=pd.read_csv(p,sep=sep,encoding=enc,low_memory=False)
                if len(df.columns)>2:return df
            except (UnicodeError, pd.errors.ParserError):pass
    raise ValueError(f'Não foi possível ler {p.name}')

def pick(cols,choices):
    return next((x for x in choices if x in cols),None)

def prepare():
    files=sorted([*DATA.glob('*.csv'),*DATA.glob('*.xlsx')])
    if not files:raise FileNotFoundError('Baixe a base Celulares subtraídos da SSP-SP e coloque o CSV/XLSX em data/.')
    frames=[]
    for p in files:
        df=read_file(p); df.columns=[norm(c) for c in df.columns]
        city=pick(df.columns,['CIDADE','CIDADE_FATO','MUNICIPIO_FATO','MUNICIPIO','CIDADE_OCORRENCIA','NOME_MUNICIPIO_CIRC'])
        nature=pick(df.columns,['RUBRICA','NATUREZA','NATUREZA_OCORRENCIA','DESCRICAO_NATUREZA'])
        bairro=pick(df.columns,['BAIRRO','BAIRRO_FATO','BAIRRO_OCORRENCIA'])
        date=pick(df.columns,['DATA_OCORRENCIA','DATA_FATO','DATA_DO_FATO','DATAOCORRENCIA','DATA_OCORRENCIA_BO'])
        hour=pick(df.columns,['HORA_OCORRENCIA','HORA_FATO','HORA_DO_FATO','HORAOCORRENCIA','HORA_OCORRENCIA_BO'])
        if not all([city,nature,bairro,date,hour]):
            raise ValueError(f'Colunas insuficientes em {p.name}. Encontradas: {list(df.columns)}. Necessárias: cidade do fato, rubrica, bairro, data e hora do fato.')
        city_s=df[city].fillna('').map(norm)
        df=df[city_s.isin(['SAO_PAULO','S_PAULO','SAOPAULO'])].copy()
        rubric=df[nature].fillna('').map(norm)
        is_roubo=rubric.str.contains(r'(^|_)ROUBO(_|$)',regex=True)
        is_furto=rubric.str.contains(r'(^|_)FURTO(_|$)',regex=True)
        df=df[is_roubo ^ is_furto].copy()
        rubric=rubric.loc[df.index]
        df['alvo']=rubric.str.contains(r'(^|_)ROUBO(_|$)',regex=True).astype(int)
        df['bairro']=df[bairro].fillna('NAO INFORMADO').map(norm).replace('', 'NAO_INFORMADO')
        dt=pd.to_datetime(df[date],errors='coerce',dayfirst=True)
        # Aceita horas como '18:30:00', objetos time e números inteiros.
        h=pd.to_numeric(df[hour].astype(str).str.extract(r'^(\d{1,2})(?=:)')[0],errors='coerce')
        h=h.where(h.between(0,23))
        df['periodo']=pd.cut(h,bins=[-1,5,11,17,23],labels=['MADRUGADA','MANHA','TARDE','NOITE']).astype(object)
        df['dia_semana']=dt.dt.dayofweek.map(dict(enumerate(['SEG','TER','QUA','QUI','SEX','SAB','DOM'])))
        # Chave do BO: múltiplos objetos de um mesmo boletim não viram amostras independentes.
        bo=pick(df.columns,['NUM_BO','NUMERO_BO','NUM_BOLETIM'])
        year=pick(df.columns,['ANO_BO','ANO_BOLETIM'])
        deleg=pick(df.columns,['NOME_DELEGACIA','DELEGACIA'])
        if bo and year and deleg:
            key=df[[bo,year,deleg]].astype(str).agg('|'.join,axis=1)
            inconsistent=df.groupby(key)['alvo'].transform('nunique')>1
            df=df[~inconsistent].copy(); key=key.loc[df.index]
            df=df.loc[~key.duplicated()].copy()
        else:
            raise ValueError(f'{p.name}: não foi possível identificar NUM_BO, ANO_BO e NOME_DELEGACIA para deduplicar boletins com segurança.')
        frames.append(df[FEATURES+['alvo']])
    all_df=pd.concat(frames,ignore_index=True).dropna(subset=['periodo','dia_semana'])
    if len(all_df)<200:raise ValueError('Menos de 200 ocorrências válidas. Confira o arquivo e o filtro geográfico.')
    if all_df.alvo.nunique()!=2:raise ValueError('A base precisa conter roubos e furtos.')
    # Amostra estratificada, sem fabricar balanceamento artificial.
    if len(all_df)>20000:
        all_df,_=train_test_split(all_df,train_size=20000,stratify=all_df.alvo,random_state=SEED)
    return all_df.reset_index(drop=True),[p.name for p in files]

class MLP(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__(); layers=[]; prev=dim
        for width in hidden:
            layers.extend([nn.Linear(prev,width),nn.ReLU(),nn.Dropout(0.2)]);prev=width
        layers.append(nn.Linear(prev,1));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).flatten()

def score(y,p):
    return {k:round(float(fn(y,p,zero_division=0) if k!='acuracia' else fn(y,p)),4) for k,fn in [('acuracia',accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]}

def train():
    df,sources=prepare()
    X_trainval,X_test,y_trainval,y_test=train_test_split(df[FEATURES],df.alvo,test_size=.15,stratify=df.alvo,random_state=SEED)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=0.15/0.85,stratify=y_trainval,random_state=SEED)
    prep=ColumnTransformer([('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),FEATURES)])
    a=prep.fit_transform(X_train);b=prep.transform(X_val);c=prep.transform(X_test)
    dense=lambda z:torch.tensor(z.toarray() if hasattr(z,'toarray') else z,dtype=torch.float32)
    a,b,c=map(dense,(a,b,c)); ya=torch.tensor(y_train.to_numpy(),dtype=torch.float32);yb=torch.tensor(y_val.to_numpy(),dtype=torch.float32)
    results={}; histories={}; states={}
    for name,hidden in CONFIGS.items():
        torch.manual_seed(SEED)
        net=MLP(a.shape[1],hidden); opt=torch.optim.Adam(net.parameters(),lr=.001); loss_fn=nn.BCEWithLogitsLoss()
        best_loss=float('inf');best_state=None;wait=0;hist={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
        for epoch in range(60):
            net.train();order=torch.randperm(len(a))
            for ix in order.split(128):
                opt.zero_grad();loss=loss_fn(net(a[ix]),ya[ix]);loss.backward();opt.step()
            net.eval()
            with torch.no_grad():
                tr=net(a);va=net(b);tl=loss_fn(tr,ya).item();vl=loss_fn(va,yb).item()
                hist['train_loss'].append(tl);hist['val_loss'].append(vl)
                hist['train_acc'].append(float(((torch.sigmoid(tr)>=.5)==ya.bool()).float().mean()))
                hist['val_acc'].append(float(((torch.sigmoid(va)>=.5)==yb.bool()).float().mean()))
            if vl<best_loss-1e-4:
                best_loss=vl;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};wait=0
            else:wait+=1
            if wait>=8:break
        net.load_state_dict(best_state);net.eval()
        with torch.no_grad():pv=(torch.sigmoid(net(b)).numpy()>=.5).astype(int)
        results[name]={**score(y_val,pv),'epocas':len(hist['train_loss']),'camadas':hidden,'val_loss':round(best_loss,4)}
        histories[name]=hist;states[name]=best_state
    # Seleção SOMENTE por validação; teste intocado até aqui.
    chosen=max(results,key=lambda n:(results[n]['f1'],results[n]['acuracia']))
    net=MLP(a.shape[1],CONFIGS[chosen]);net.load_state_dict(states[chosen]);net.eval()
    with torch.no_grad(): probs=torch.sigmoid(net(c)).numpy()
    pred=(probs>=.5).astype(int)
    test_metrics=score(y_test,pred);cm=confusion_matrix(y_test,pred,labels=[0,1]).tolist()
    joblib.dump(prep,OUT/'preprocessador.joblib');torch.save({'state_dict':states[chosen],'input_dim':a.shape[1],'hidden':CONFIGS[chosen]},OUT/'modelo.pt')
    h=histories[chosen]
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].plot(h['train_loss'],label='Treino');axs[0].plot(h['val_loss'],label='Validação');axs[0].set_title('Perda');axs[1].plot(h['train_acc'],label='Treino');axs[1].plot(h['val_acc'],label='Validação');axs[1].set_title('Acurácia')
    for ax in axs:ax.set_xlabel('Época');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'curvas.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1],['Furto','Roubo']);ax.set_yticks([0,1],['Furto','Roubo']);ax.set_xlabel('Previsto');ax.set_ylabel('Real');ax.set_title('Matriz de confusão — teste')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(OUT/'matriz_confusao.png',dpi=160);plt.close(fig)
    report={'fonte':'https://www.ssp.sp.gov.br/estatistica/consultas','arquivos':sources,'registros':len(df),'classes':{'furto':int((df.alvo==0).sum()),'roubo':int((df.alvo==1).sum())},'divisao':{'treino':len(X_train),'validacao':len(X_val),'teste':len(X_test)},'atributos':FEATURES,'arquiteturas':results,'selecionada':chosen,'teste':test_metrics,'matriz_confusao':cm,'periodo':'Conferir o período dos arquivos originais da SSP-SP'}
    (OUT/'resultados.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False));return report

if __name__ == "__main__":
    print("Iniciando treinamento das MLPs...")
    train()