# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:14:41 2023
@author: Freddy J. Orozco R.
@Powered: WinStats.
"""

import streamlit as st
#import hydralit_components as hc
import datetime
import base64
import pandas as pd
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import mplsoccer
from mplsoccer import Pitch, VerticalPitch, FontManager
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib import colors as mcolors
import requests
from PIL import Image
from matplotlib.patches import Rectangle
import math
import io

st.header("REGISTER DATA")

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'Nombre': ['Juan', 'María', 'Pedro', 'Ana', 'Luis'],
    'Edad': [25, 30, 35, 28, 40],
    'Ciudad': ['Madrid', 'Barcelona', 'Sevilla', 'Valencia', 'Bilbao'],
    'Salario': [30000, 35000, 40000, 32000, 38000]
})

# Función para crear el contenido del archivo Excel para una fila
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


# Función para generar el enlace de descarga
def get_download_link(row):
    val = to_excel(pd.DataFrame([row]))
    b64 = base64.b64encode(val).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{row["Nombre"]}_datos.xlsx">📥</a>'

# Agregar la columna de enlaces de descarga al DataFrame
df['Descargar'] = df.apply(lambda row: get_download_link(row), axis=1)

# Mostrar el DataFrame con los enlaces de descarga
st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
st.write(df)

# Estilo CSS para mejorar la apariencia de la tabla

st.write("AA")
# Botón para descargar todo el DataFrame
st.write("Descargar todo el DataFrame:")
full_excel_data = to_excel(df)
st.download_button(
    label="Descargar todo el DataFrame",
    data=full_excel_data,
    file_name="todos_los_datos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="download_all"
)
