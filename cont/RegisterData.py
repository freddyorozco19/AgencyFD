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

def get_table_download_link(df, filename):
    """Genera un link para descargar los datos en formato Excel"""
    val = to_excel(df)
    b64 = base64.b64encode(val).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.xlsx">Descargar</a>'

# Generar la tabla HTML con botones de descarga integrados
table_html = "<table>"
table_html += "<tr><th>Nombre</th><th>Edad</th><th>Ciudad</th><th>Salario</th><th>Descargar</th></tr>"

for index, row in df.iterrows():
    table_html += "<tr>"
    for col in df.columns:
        table_html += f"<td>{row[col]}</td>"
    
    # Agregar el botón de descarga como un enlace en la última columna
    download_link = get_table_download_link(pd.DataFrame([row]), f"{row['Nombre']}_datos")
    table_html += f"<td>{download_link}</td>"
    
    table_html += "</tr>"

table_html += "</table>"

# Estilo CSS para la tabla
st.markdown("""
<style>
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
}
</style>
""", unsafe_allow_html=True)

# Mostrar la tabla HTML
st.markdown(table_html, unsafe_allow_html=True)
