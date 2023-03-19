from bs4 import BeautifulSoup
import requests
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize  as solver
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

class Financias:
    def __init__(self, acoes, inicio, fim):    
        self.__acoes = acoes    
        self.__inicio = inicio    
        self.__fim = fim    
        self.__dados  = self.__tratar_dados()

    def __puxar_dados(self, s):
        yf.pdr_override()
        data = pdr.get_data_yahoo(s, start=self.__inicio, end=self.__fim)
        dolar = self.__pegar_dolar()
        data['Close'] = data['Close'].apply(lambda x: x*dolar)
        data['Retorno'] = data['Close'].pct_change()

        return data

    def __tratar_dados(self):
        if isinstance(self.__acoes, list):
            print(self.__acoes)
            df = list()
            for s in self.__acoes:
                dados = self.__puxar_dados(s)
                dados.rename(columns={'Close':s}, inplace=True)
                dados = dados[[s, 'Retorno']]
                df.append(dados)
            return pd.concat(df, axis=1)
        else:
            dados = self.__puxar_dados(self.__acoes)
            dados.rename(columns={'Close':self.__acoes}, inplace=True)
            dados = dados[[self.__acoes, 'Retorno']]
            return dados

    
    def fronteira(self):
        prec = self.__dados[self.__acoes]
        ri = self.__dados['Retorno']
        mi = ri.mean()*len(self.__dados)
        sigma = ri.cov()*len(self.__dados)
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
    
    def monte_carlo(self, fim):
        prec = self.__dados.drop('Retorno', axis=1)
        dolar = self.__pegar_dolar()
        prec = prec.apply(lambda x: x/dolar)
        
        ri = self.__dados['Retorno']
        mi = ri.mean()
        sigma = ri.std()
        x = np.zeros((len(prec),1000))
        for j in range(1000):
            for i in range(len(prec)):
                x[i,j] = np.random.normal(mi, sigma)
        dados = pd.DataFrame(dict(dado=['Retorno', 'Retorno', 'Volatidade', 'Volatidade'], simulado=['Real', 'Simulado']*2,
                                    valor=[f'{mi*len(prec):.3%}', f'{x.mean()*len(prec):.3%}', f'{sigma*len(prec):.2%}', f'{x.std()*len(prec):.2%}']))
        num = 100
        eixo = self.__range_date(fim)
        dias =len(eixo) 
        aleat = np.zeros((dias, num))
        simul = np.zeros((dias, num))
        print(len(prec))
        for j in range(num):
            for i in range(dias):
                aleat[i,j] = np.random.normal(mi, np.sqrt(i)*sigma/np.sqrt(dias))
                simul[i, j] = prec[-1:].values[0]+aleat[i,j]
        
        prec['tipo']='real'
        
       
        otimista = [simul[i,:].max() for i in range(dias)]
        pessimista = [simul[i,:].min() for i in range(dias)]
        realista = [np.median(simul[i,:]) for i in range(dias)]

        futuro_otimista = pd.DataFrame({'Date': eixo, f'{self.__acoes}':otimista, 'tipo':['otimista']*dias}).set_index('Date')
        futuro_pessimista = pd.DataFrame({'Date': eixo, f'{self.__acoes}':pessimista, 'tipo':['pessimista']*dias}).set_index('Date')
        futuro_realista = pd.DataFrame({'Date': eixo, f'{self.__acoes}':realista, 'tipo':['Estabilidade']*dias}).set_index('Date')
        predicao = pd.concat([prec, futuro_otimista, futuro_pessimista, futuro_realista])
        predicao[self.__acoes] = predicao[self.__acoes].apply(lambda x: x*dolar)

        


        fig = px.line(data_frame=predicao, y=self.__acoes, color='tipo', title=f'<b>Série Histórica<br> {self.__acoes}', labels={self.__acoes:'R$'})
        fig.update_layout(title_x=0.35)
        
        return dados, fig
    
    def __range_date(self, fim):
        start_date = self.__fim
        end_date = fim
        days = (end_date - start_date).days + 1
        dias = list()
        for i in range(days):
            date = start_date + dt.timedelta(days=i) if i>0 else start_date
            dias.append(date)
        return dias
    
    def __pegar_dolar(self):
        url = ' https://economia.awesomeapi.com.br/last/USD-BRL'
        response = requests.get(url)
        dolar = float(response.json()['USDBRL']['bid'])
        return dolar



        
       
       

    


