from pmdarima.arima import auto_arima
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pylatex import Document, LongTable, MultiColumn
from datetime import datetime, timedelta
import os, os.path
from pylatex import Document, Section, LongTable, MultiColumn, Package, Command
from datetime import datetime, timedelta
from pylatex.utils import NoEscape

class ModeloARIMA():
    def __init__(self, datos: str) -> None:
        self.df = pd.read_excel(datos)
        self.variables = self.df.columns.tolist()
        self.ruta_escritorio = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        vars = []
        for var in self.variables:
            try:
                self.df[var].astype("int64")
                vars.append(var)
            except:
                pass
        self.vars_numericas = vars

    def hallar_parametros(self):
        modelo_var = {var : (0,0,0) for var in self.vars_numericas}
        for var in self.vars_numericas:
            modelo = auto_arima(self.df[var], max_p=100, max_d=5, max_q=100)
            modelo_var[var] = modelo.order
        return modelo_var
    
    def predecir_ARIMA(self, tiempo):
        dic_parametros = self.hallar_parametros()
        dic_predicciones = {var: 0 for var in self.vars_numericas if var != "Año"}
        for var in dic_predicciones.keys():
            modelo = ARIMA(self.df[var], order=dic_parametros.get(var))
            resultado = modelo.fit()
            # Formatear cada valor de la predicción a dos decimales
            dic_predicciones[var] = [round(x, 2) for x in resultado.forecast(steps=tiempo)]
        return dic_predicciones
    

    def generar_latex(self, tiempo):
        predicciones = self.predecir_ARIMA(tiempo)
        longitud_lista = len(next(iter(predicciones.values())))
        # Crear la lista de listas donde cada sublista contiene el i-ésimo elemento de cada lista del diccionario
        arima = [[predicciones[key][i] for key in predicciones] for i in range(longitud_lista)]
        marca_temp = datetime.now().strftime("%d-%m-%Y%H%M%S")
        meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

        # Crear un nuevo documento LaTeX con orientación horizontal
        geometry_options = {
            "margin": "1in",
            "landscape": True,
            "headheight": "2in",  # Ajusta esto según sea necesario
            "headsep": "0.5in",   # Espacio entre el encabezado y el texto
        }
        doc = Document(geometry_options=geometry_options)

        # Añadir paquetes necesarios para imágenes y encabezados personalizados
        doc.packages.append(Package('graphicx'))  # Para incluir imágenes
        doc.packages.append(Package('fancyhdr'))  # Para encabezados personalizados
        doc.packages.append(Package('booktabs'))

        # Configurar la página con encabezado personalizado
        doc.preamble.append(Command('pagestyle', 'fancy'))
        doc.preamble.append(Command('fancyhf', ''))  # Limpia el encabezado y pie de página actuales
        #doc.preamble.append(Command('setlength', NoEscape(r'\headheight'), '27pt'))  # Ajusta la altura del encabezado
        # doc.preamble.append(Command('lhead', NoEscape(r'\includegraphics[height=2cm]{logo.png}')))
        
        doc.preamble.append(NoEscape(r'\title{Predicción IPC para los próximos %s meses}' % tiempo))
        doc.preamble.append(NoEscape(r'\date{}'))  # No mostrar fecha
        doc.append(NoEscape(r'\maketitle'))

        header_cells = ["Año", "Mes"] + list(predicciones.keys())

        # Crea una sección sin numerar para la tabla
        with doc.create(Section('Predicciones detalladas', numbering=False)) as section:
            # Crea un entorno de tabla larga con la cantidad correcta de columnas
            with section.create(LongTable("|l " + "r|" * (len(header_cells)-1))) as table:
                # Añade una línea horizontal en la parte superior de la tabla
                table.add_hline()
                # Agrega una sola vez el encabezado de la tabla con MultiColumn
                table.add_row(header_cells)
                table.add_hline()

            # Agregar las filas de datos con la marca temporal correspondiente
            mes_inicio = int(datetime.now().strftime("%m"))
            año_inicio = int(datetime.now().strftime("%Y"))
            años = []
            # Agregar meses faltantes del año en curso
            for i in range(12 - mes_inicio):
                años.append(año_inicio)
            # Agregar meses de años que se completan
            for i in range(1, int((tiempo - 12 + mes_inicio) / 12) + 1):
                for j in range(12):
                    años.append(año_inicio + i)
            # Agregar residuo de la división como meses de año no completado en estimación
            for i in range((tiempo - 12 + mes_inicio)%12):
                años.append(año_inicio + int((tiempo - 12 + mes_inicio) / 12) + 1)

            # Hacer lista de meses
            meses_tabla = [meses[(mes_inicio + i - 1) % 12] for i in range(1, tiempo + 1)]

            for i in range(tiempo):
                # Formatear las predicciones para la fila actual
                formatted_predictions = [años[i], meses_tabla[i]] + [valor for valor in arima[i]]

                # Agregar la fila con el año, mes y los valores de predicción formateados
                table.add_row(formatted_predictions)

            table.add_hline()

             # Añadir las tablas de resumen al documento
            for var, summary_latex in summaries.items():
                doc.append(NoEscape(r'\subsection*{%s}' % var))
                doc.append(NoEscape(summary_latex))
    
        os.environ['PATH'] += os.pathsep + 'C:\\Users\\mchinchilla\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64'
        os.environ['PATH'] += os.pathsep + 'C:\Strawberry\perl\bin'
        # Guardar el documento
        doc.generate_pdf(os.path.join(self.ruta_escritorio, "Programa Muestreo", "Modelos ARIMA", 
                                      "Predicciones", f"Prediccion_IPC_{marca_temp}"), clean_tex=False, compiler='pdflatex')