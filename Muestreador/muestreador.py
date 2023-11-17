# Importar librerías
import pandas as pd
from scipy.stats import t, norm
import numpy as np
import logging

class MuestreoINE:
    def __init__(self, tamaño_poblacion: int=0, varianza: np.array=np.zeros, costos: np.array=np.zeros):
        self.poblacion = tamaño_poblacion
        self.varianza = varianza
        self.costos = costos
    # Métodos para cálculo de tamaño de muestra

    # SRS
    def tamaño_srs(self, error, confianza, varianza) -> int:
        z = norm.ppf(1- ((1 - confianza) / 2))
        n = (varianza * z**2) / error**2
        return n / (1 + n / self.poblacion)

    # Cálculo de tamaño de muestra por estrato
    def tamaño_estratificado(self, error, confianza, index_kish, proporciones: list) -> dict:
        # Las proporciones se pasan en una lista en el siguiente orden: 
        # Proporción óptima, constante, uniforme
        try:
            tamaños = {"Óptimo": 0, "Kish": 0, "Proporcional": 0, "Uniforme": 0}
            # optima
            numerador = proporciones[0] * np.sqrt(self.varianza) / np.sqrt(self.costos)
            denominador = np.sum(numerador)
            tamaños["Óptimo"] = self.tamaño_srs(error, confianza) * numerador/denominador
            
        # Kish
            numerador = np.sqrt(1 / np.len(proporciones[0])**2 + index_kish * proporciones[0]**(2))
            denominador = np.sum(numerador)
            tamaños["Kish"] = self.tamaño_srs(error, confianza) * numerador / denominador

            # proporcional 
            N_h = self.poblacion * proporciones[1]
            k = self.tamaño_srs(error, confianza) / self.poblacion
            tamaños["Proporcional"] = k * N_h

            # uniforme
            tamaños["Uniforme"] = self.tamaño_srs(error, confianza) / np.len(proporciones[2]) * np.ones_like(proporciones[2]) 

            return tamaños
        except Exception as e:
            logging.error(f"Error al calcular el tamaño de las muestras: {e}")

    # Cálculo de estimadores usando SRS. Se recibe por parámetro una serie de pandas para la cual se calcularán los estadísticos
    def estimadores_SRS(self, variable: pd.Series) -> dict:
        # Leer datos usando parámetro variable para indicar el nombre de la variable aleatoria para los cálculos
        try:
            tamaño_muestra = len(variable) - 1
            fcp = 1 - ( tamaño_muestra / self.poblacion)
            # Cálculo de estimadores
            varianza_poblacional = variable.var()
            total_poblacional = self.poblacion * variable.mean()
            media_poblacional = total_poblacional / self.poblacion
            # Cálculo de errores estándar
            e_total_poblacional = self.poblacion * np.sqrt( fcp* varianza_poblacional / tamaño_muestra)
            e_media_poblacional = np.sqrt(fcp * varianza_poblacional / tamaño_muestra)

            estimadores_y_errores = {"Estimadores" : [total_poblacional, media_poblacional], "Errores": [e_total_poblacional, e_media_poblacional]}
            return estimadores_y_errores
        
        except Exception as e:
            logging.error(f"Error al calcular estimadores con SRS: {e}")

    # Método para calcular estimadores usando muestreo estratificado. Se asume que se recibe una lista con los estratos. 
    # Faltaría adaptar esto a que se reciba un df con los datos por estrato y se itere en él 

    # Intervalos de confianza
    def media_IC_z(self, confianza, columna, datos: pd.DataFrame):
        z = norm.ppf(0.5 + confianza / 2)
        media = datos[columna].mean()
        n = len(datos)
        error = z * np.squeeze(self.varianza) / np.sqrt(n)
        return (media - error, media +  error)
    
    # Estimadores en muestreo estratificado
    def media_estratificado(self, columna_pesos, columna_va, datos) -> float:
        datos["w_jy_j"] = datos[columna_pesos] * datos[columna_va]
        return datos["w_jy_j"].sum() / datos[columna_pesos].sum()
    

    # Calcular un intervalo de confianza con confianza dada para media con muestreo estratificado
    def media_IC_estratificado_z(self, confianza: float, tamaño_estratos: list, tamaño_muestras: list,
                                  columna_pesos: str, datos: pd.DataFrame, nombre_columna_estrato: str, nombre_col_va: str):
        # Validación de los tamaños de los estratos y muestras
        if len(tamaño_estratos) != len(tamaño_muestras):
            raise ValueError("Los tamaños de los estratos y las muestras deben coincidir.")
        
        if len(tamaño_estratos) == 0 or  len(tamaño_muestras) == 0:
            raise ValueError("Ingrese datos válidos. La lista es vacía.")
        
        n_total = sum(tamaño_muestras)
        N_total = sum(tamaño_estratos)
        
        varianzas = datos.groupby(nombre_columna_estrato)[nombre_col_va].var(ddof=1).fillna(0).tolist()

        v = 0
        for i in range(len(varianzas)):
            n_estrato = tamaño_muestras[i]
            N_estrato = tamaño_estratos[i]
            varianza_estrato = varianzas[i]
            v += n_total / n_estrato * N_estrato**2 / N_total**2 * varianza_estrato**2
        
        z = norm.ppf(0.5 + confianza / 2)
        media = self.media_estratificado(columna_pesos, nombre_col_va, datos)
        error = z * np.sqrt(v / n_total)

        return (media - error, media + error)
    
    # Intervalo de confianza para proporción con muestreo estratificado
    def proporcion_IC_estratificado(self, confianza: float, tamaño_estratos: list, tamaño_muestras: list, proporciones: list):
        
        # Validación de los tamaños de los estratos y muestras
        if len(tamaño_estratos) != len(tamaño_muestras):
            raise ValueError("Los tamaños de los estratos y las muestras deben coincidir.")
        
        if len(tamaño_estratos) == 0 or  len(tamaño_muestras) == 0:
            raise ValueError("Ingrese datos válidos. La lista es vacía.")
        
        N_total = sum(tamaño_estratos)
        n_total = sum(tamaño_muestras)

        proporcion_estimador = 0
        for i in range(len(tamaño_estratos)):
            proporcion_estimador += tamaño_estratos[i] / N_total * proporciones[i]
        
        varianzas = []
        for i in range(len(tamaño_muestras)):
            if tamaño_muestras[i] != 1:
                varianza = tamaño_muestras[i] / (tamaño_muestras[i] - 1) * proporciones[i] * (1 - proporciones[i])
            else:
                varianza = 0
            varianzas.append(varianza)

        v = 0
        for i in range(len(varianzas)):
            n_estrato = tamaño_muestras[i]
            N_estrato = tamaño_estratos[i]
            varianza_estrato = varianzas[i]
            v += n_total / n_estrato * N_estrato**2 / N_total**2 * varianza_estrato**2

        z = norm.ppf(0.5 + confianza / 2)
        error = z * np.sqrt(v/n_total)

        return (proporcion_estimador -  error, proporcion_estimador + error)