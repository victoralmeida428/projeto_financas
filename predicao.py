
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
from pesos import Financias

st.set_page_config(page_title='Finance Project', page_icon=":chart_with_upwards_trend:")
def puxar_dados(s, start, end):
   yf.pdr_override()
   data = pdr.get_data_yahoo(s, start=start, end=end)
   data['Retorno'] = data['Close'].pct_change()
   return data


tickers = pd.read_csv('nasdaq_screener_1678405255104.csv')
tbr = pd.read_csv('BR.csv')
tbr = tbr.Symbol.tolist()
tickers = tickers.Symbol.tolist()
alltickers = tbr+tickers

st.sidebar.title(':green[**Mercado Financeiro**]')
inicio = st.sidebar.date_input('Inicio: ', )
fim = st.sidebar.date_input('Fim: ', max_value=dt.datetime.now())

symbol = st.sidebar.multiselect('Escolha as ações',options=alltickers)
radio_opt = ['Série Histórica', 'Retorno Financeiro', 'Fronteira eficiente']
serie = st.sidebar.selectbox('Modelo (R$)', options=radio_opt)
dfs = list()

def serie_model(data, serie, symbols=''):
  if serie == 'Série Histórica':
      fig = px.line(data_frame=data, y='Close', 
                    color='Tickers', title=f'<b>Série Histórica<br> {symbols}',
                    labels=dict(Close='Close - R$'))
      fig.update_layout(title_x=0.35)
      return fig
  if serie == 'Retorno Financeiro':
     
     fig = px.line(data_frame=data, y='Retorno', color='Tickers',
                    title=f'<b>Retorno financeiro<br> {symbols}',
                    labels=dict(Close='Retorno'))
     fig.update_layout(title_x=0.35)
     return fig
  if serie == 'Fronteira eficiente':
    if len(symbol)>1:
      fin = Financias(symbol, inicio, fim)
      fig, po = fin.fronteira()
      table = pd.DataFrame(dict(Ação=symbol, Participação=po))
      table['Participação'] = table['Participação'].apply(lambda x: f'{x:.2%}')
            
      st.table(table)
      st.write(fig)
    else:
      st.write('Favor escolher pelo menos 2 ativos')

if len(symbol)>1:
  for s in symbol:
      data = puxar_dados(s, start=inicio, end=fim)
      data['Tickers'] = s
      dfs.append(data)
  df = pd.concat(dfs, ignore_index=False)

  st.write(serie_model(df, serie, symbol))
elif len(symbol) == 1:
  data = puxar_dados(symbol, start=inicio, end=fim)
  data['Tickers'] = symbol[0]
  st.write(serie_model(data, serie, symbol))
else:
  st.title(':green[**Project Finance**]')
  st.write('Se você é um investidor, provavelmente já se perguntou qual é a melhor maneira de avaliar um ativo antes de investir nele. Felizmente, existem várias ferramentas que podem ajudá-lo nessa tarefa. Uma delas é a análise fundamentalista, que envolve a avaliação das condições financeiras e do desempenho da empresa em questão. Através dessa ferramenta, é possível obter informações sobre o fluxo de caixa, receita, lucro líquido e outros fatores que influenciam o valor do ativo.')



  









