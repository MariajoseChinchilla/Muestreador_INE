# Importar librerias
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm
import numpy as np

class ARIMAPruebas():
    def __init__(self, alpha: float, datos: str):
        self.alpha = alpha
        self.datos = pd.read_excel(datos)

    # Aplicar el test ADF suponiendo que los datos de entreno están en Excel
    def test_ADF(self,  var_aleatoria: str, rezagos: int=0):
        serie = self.datos[var_aleatoria]
        for _ in range(rezagos):
            serie = serie.diff().dropna()
        return adfuller(serie)
      
    # Realizar test ADF y devuelve un booleano para estacionalidad. Hace prueba de hipótesis.
    def es_estacionaria(self, var_aleatoria, rezagos: int=0):
        resultado_adf = self.test_ADF(var_aleatoria, rezagos)
        p_valor = resultado_adf[1]
        
        if p_valor < self.alpha:
            return True
        else:
            return False
    
    def calcular_d(self, var_aleatoria):
        diferenciaciones = 0
        es_estacionaria = self.es_estacionaria(var_aleatoria, diferenciaciones)
        
        while not es_estacionaria:
            diferenciaciones += 1
            es_estacionaria = self.es_estacionaria(var_aleatoria, diferenciaciones)
        
        return diferenciaciones
    
    # Se aproxima el intervalo de confianza para la función de correlación parcial suponiendo se tienen suficientes datos
    # para aproximar el error estándar como 1/raíz(N).
    def calcular_max_p(self, var_aleatoria: str):
        p = 0
        z = norm.ppf(1-self.alpha/2)
        N = len(self.datos[var_aleatoria])
        pacf_ic = (-1 * z / np.sqrt(N), z / np.sqrt(N))
        for correlacion in pacf(self.datos[var_aleatoria]):
            if correlacion < pacf_ic[0] or correlacion > pacf_ic[1]:
                p += 1
        return p
    
 
    def calcular_max_q(self, var_aleatoria: str):
        z = norm.ppf(1-self.alpha/2)
        N = len(self.datos[var_aleatoria])
        q = 0 
        acf_ic = (-1 * z / np.sqrt(N), z / np.sqrt(N))
        for correlacion in acf(self.datos[var_aleatoria]):
            if correlacion < acf_ic[0] or correlacion > acf_ic[1]:
                q += 1
        return q