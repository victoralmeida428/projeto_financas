
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize  as solver

class Financias:
    def __init__(self, acoes, inicio, fim):    
        self.__acoes = acoes    
        self.__inicio = inicio    
        self.__fim = fim    
        dados = self.__tratar_dados()
        self.__dados = dados

    def __puxar_dados(self, s):
        yf.pdr_override()
        data = pdr.get_data_yahoo(s, start=self.__inicio, end=self.__fim)
        data['Retorno'] = data['Close'].pct_change()
        return data

    def __tratar_dados(self):
        df = list()
        for s in self.__acoes:
            dados = self.__puxar_dados(s)
            dados.rename(columns={'Close':s}, inplace=True)
            dados = dados[s]
            df.append(dados)
        return pd.concat(df, axis=1)
    
    def fronteira(self):
        prec = self.__dados
        ri = prec/prec.shift(1)-1
        mi = ri.mean()*252
        sigma = ri.cov()*252
        vet_r= list()
        vet_vol = list()
        def estatis_port(peso, mi, sigma):
            peso = np.array(peso)
            ret_ot = np.sum(peso*mi)
            risco_ot = np.sqrt(np.dot(peso.T, np.dot(sigma, peso)))
            return ret_ot, risco_ot

        for i in range(2000):
            #----PESOS DA ALOCAÇÃO DOS INVESTIMENTOS -----
            w = np.random.random(len(self.__acoes))
            w = w/np.sum(w)
            #-----RETORNO E RISCO DA CARTERIA-------
            retorno, risco = estatis_port(w, mi, sigma)
            ###############################################
            vet_r.append(retorno)
            vet_vol.append(risco)

        pesos = pd.DataFrame(dict(retorno = vet_r, risco = vet_vol))

        def f_obj(peso):
            return np.sqrt(np.dot(peso.T, np.dot(sigma, peso)))

        x0 = np.array([1.0/len(self.__acoes) for x in range(len(self.__acoes))])
        bounds = tuple((0,1) for x in range(len(self.__acoes)))
        faixa_ret = np.arange(pesos.retorno.min(), pesos.retorno.max(), 0.001)

        risk = list()
        for i in faixa_ret:
            constraints = [{'type':'eq','fun': lambda x: np.sum(x)-1},
                        {'type':'eq','fun': lambda x: np.sum(x*mi)-i}]
            outcomes = solver.minimize(f_obj, x0, constraints=constraints, bounds=bounds, method='SLSQP')
            risk.append(outcomes.fun)

        for i in faixa_ret:
            constraints = [{'type':'eq','fun': lambda x: np.sum(x)-1}]
            outcomes = solver.minimize(f_obj, x0, constraints=constraints, bounds=bounds, method='SLSQP')
            risk.append(outcomes.fun)

        ret_ot, vol_ot = estatis_port(outcomes['x'], mi, sigma)

        fig = px.scatter(data_frame=pesos,x='risco', y='retorno', opacity=0.3, title='<b>Otimização de portifólio<br>Fronteira Eficiente')
        fig.add_trace(go.Scatter(x=risk, y=faixa_ret, name='Linha otimizada', line={'color':'green'}, mode='lines'))
        fig.add_annotation(text=f'<b>Ponto Ótimo<br>({vol_ot:.2f},{ret_ot:.2f})', 
                        xref='x', yref='y', ax=-60, ay=-40,
                        x=vol_ot, y=ret_ot, showarrow=True, align='left', arrowcolor='green', font=dict(color='green', size=12))
        fig.update_layout(title_x=0.5)

        pesos_otimos = outcomes['x']
        return fig, pesos_otimos
    
