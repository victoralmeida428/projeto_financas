import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from skfuzzy import control as clt
import skfuzzy as fuzz
import numpy as np
import pandas as pd

df = []


yf.pdr_override()
y_symbols = ['PETR4.SA', 'ITUB4.SA', 'ABEV3.SA', 'GGBR4.SA']
inicio = dt.datetime(2015,3,1)
fim = dt.datetime.now()
data = pdr.get_data_yahoo(y_symbols, start=inicio, end=fim)

for item in y_symbols:
    vol_info = data['Volume'][item].describe()

    cotacao_preco = data['Close'][item].iloc[-1]
    cotacao_volume = data['Volume'][item].iloc[-1]

    print(f'Preco = {cotacao_preco}\nVolume: {cotacao_volume}')

    preco_info = data['Close'][item].describe()
    

    preco = clt.Antecedent(np.arange(data['Close'][item].min(),data['Close'][item].max(),1), 'preço')
    vol = clt.Antecedent(np.arange(data['Volume'][item].min(),data['Volume'][item].max(),1e5), 'volume')
    dec = clt.Consequent(np.arange(data['Volume'][item].min(),data['Volume'][item].max(),1e5), 'decisão')

    dp_preco = data['Close'][item].std()
    preco['barato'] = fuzz.gaussmf(preco.universe,data['Close'][item].quantile(0.25),dp_preco)
    preco['ideal'] = fuzz.gaussmf(preco.universe,data['Close'][item].quantile(.5),dp_preco)
    preco['caro'] = fuzz.gaussmf(preco.universe,data['Close'][item].quantile(.75),dp_preco)
    dp_vol = data['Volume'][item].std()
    vol['baixo'] = fuzz.gaussmf(vol.universe,data['Volume'][item].quantile(0.25), dp_vol)
    vol['ideal'] = fuzz.gaussmf(vol.universe,data['Volume'][item].quantile(.5), dp_vol)
    vol['alto'] = fuzz.gaussmf(vol.universe,data['Volume'][item].quantile(.75), dp_vol)

    dec['comprar'] = fuzz.gaussmf(dec.universe,data['Volume'][item].quantile(0.25), dp_vol)
    dec['manter'] = fuzz.gaussmf(dec.universe,data['Volume'][item].quantile(.5), dp_vol)
    dec['vender'] = fuzz.gaussmf(dec.universe,data['Volume'][item].quantile(.75), dp_vol)




    regra1 = clt.Rule(preco['barato'] & vol['baixo'], dec['comprar'])
    regra2 = clt.Rule(preco['barato'] & vol['alto'], dec['comprar'])
    regra3 = clt.Rule(preco['ideal'] & vol['baixo'], dec['comprar'])
    regra4 = clt.Rule(preco['ideal'] & vol['ideal'], dec['manter'])
    regra5 = clt.Rule(preco['ideal'] & vol['alto'], dec['vender'])
    regra6 = clt.Rule(preco['barato'] & vol['alto'], dec['vender'])

    decisao_clt = clt.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6])
    decisao = clt.ControlSystemSimulation(decisao_clt)

    def IndFzy(entrada):
        decisao.input['preço'] = entrada[0]
        decisao.input['volume'] = entrada[1]

        decisao.compute()

        return (decisao.output['decisão'])

    res1 = IndFzy([cotacao_preco,cotacao_volume])
    dec.view(sim=decisao)
    plt.title(item)

    print(res1)

    mval = []

    for t in dec.terms:
        s = np.interp(res1, dec.universe, dec[t].mf)
        mval.append([t, s, item])

    print(mval)

    df.append(mval)

    mval = pd.DataFrame(mval)
    ind_max = mval[1].idxmax()

    print(f'--------------------------\n'
            f'DECISÃO FINAL \n'
            f'{mval[0][ind_max]} \n'
            f'------------------------')

    plt.show()

df = pd.DataFrame(df)
comprarAcoes = []
venderAcoes = []
manterAcoes = []

teste = pd.DataFrame(df[0])
print(teste)




