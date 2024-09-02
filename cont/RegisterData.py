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
def to_excel(row):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame([row]).to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
    return output.getvalue()

# Función para generar el botón de descarga
def get_download_button(row):
    excel_data = to_excel(row)
    return st.download_button(
        label="📥",
        data=excel_data,
        file_name=f"{row['Nombre']}_datos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Agregar la columna de botones de descarga al DataFrame
df['Descargar'] = df.apply(lambda row: get_download_button(row), axis=1)

# Mostrar el DataFrame con los botones de descarga
st.dataframe(df)
