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

st.header("M Analysis")
df = pd.read_excel('Data/Players_Gral.xlsx')
df['Categoria'] = df['Categoria'].astype(str)

filmenu01, filmenu02, filmenu03 = st.columns(3)
with filmenu01:
  PriorityList = df['Priority'].drop_duplicates().tolist()
  PriorityList.insert(0, "All")
  PrioritySel = st.selectbox('Choose priority:', PriorityList)
  df_bkfil01 = df
  if PrioritySel == 'All':
    df = df_bkfil01
  else:
    df = df[df['Priority'] == PrioritySel].reset_index(drop=True)
st.dataframe(df)

# Crea un DataFrame de ejemplo
data = {'Name': ['John', 'Jane', 'Bob'],
        'Link': ['https://www.example.com/john', 'https://www.example.com/jane', 'https://www.example.com/bob']}
df2 = pd.DataFrame(data)

# Muestra el DataFrame en Streamlit
st.title("DataFrame con Hipervínculos")
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
    return f'data:application/octet-stream;base64,{b64}'

# Agregar la columna de enlaces de descarga al DataFrame
df2['Descargar'] = df2.apply(lambda row: get_download_link(row), axis=1)
# Configurar la columna de descarga como un hipervínculo
column_config = {
    "Descargar": st.column_config.LinkColumn(
        "Descargar",
        display_text="Descargar",
        help="Haz clic para descargar los datos de esta fila",
    )
}
# Muestra el DataFrame con la columna de hipervínculos
st.dataframe(df2,
    column_config=column_config,
    hide_index=True,
    use_container_width=True)
