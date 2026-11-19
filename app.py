import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import torch
from modelo import MLP, OUT, norm

st.set_page_config(page_title='Classificador de Ocorrências SP',page_icon='🛡️')
st.title('Classificador de Ocorrências — São Paulo')
st.write('Classificação de uma ocorrência já registrada ou hipotética. **Não prevê a ocorrência de crimes.**')
if not (OUT/'modelo.pt').exists() or not (OUT/'preprocessador.joblib').exists():
    st.error('Modelo não encontrado. Execute `python modelo.py` com os dados reais da SSP-SP antes de abrir a aplicação.')
    st.stop()
pre=joblib.load(OUT/'preprocessador.joblib')
ck=torch.load(OUT/'modelo.pt',map_location='cpu',weights_only=True)
net=MLP(ck['input_dim'],ck['hidden']);net.load_state_dict(ck['state_dict']);net.eval()
known=list(pre.named_transformers_['cat'].named_steps['onehot'].categories_[0])
bairro=st.selectbox('Bairro do fato',known)
periodo=st.selectbox('Período do fato',['MADRUGADA','MANHA','TARDE','NOITE'])
dia=st.selectbox('Dia da semana',['SEG','TER','QUA','QUI','SEX','SAB','DOM'])
if st.button('Classificar ocorrência',type='primary'):
    row=pd.DataFrame([{'bairro':bairro,'periodo':periodo,'dia_semana':dia}])
    x=pre.transform(row);x=torch.tensor(x.toarray() if hasattr(x,'toarray') else x,dtype=torch.float32)
    with torch.no_grad():roubo=torch.sigmoid(net(x)).item()
    st.subheader('Classe estimada: '+('Roubo' if roubo>=.5 else 'Furto'))
    st.write(f'Roubo: **{roubo:.1%}** | Furto: **{1-roubo:.1%}**')
    st.caption('Probabilidades do classificador para o tipo de uma ocorrência, não taxas de risco criminal.')
