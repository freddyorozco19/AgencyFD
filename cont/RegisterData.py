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
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder
import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.header("REGISTER DATA")

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



# Botón para descargar todo el DataFrame
#st.write("Descargar todo el DataFrame:")
#full_excel_data = to_excel(df)
#st.download_button(    label="Descargar todo el DataFrame",data=full_excel_data,file_name="todos_los_datos.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="download_all")

df = pd.DataFrame(
        {"Site": "DuckDuckGo Google Bing".split(),
        "URL": "https://duckduckgo.com/ https://www.google.com/ https://www.bing.com/".split()}
    )

gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_column("URL",
                    headerName="URL",
                    cellRenderer=JsCode(
                        """
                        function(params) {
                            return '<a href=' + params.value + '> 🖱️ </a>'
                            }
                        """))

gridOptions = gb.build()
AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)

# Define los alcances necesarios para la API de Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

def read_from_sheets(spreadsheet_id, range_name):
    """Lee datos de Google Sheets y devuelve un DataFrame usando una cuenta de servicio."""
    creds = Credentials.from_service_account_file('cont/winstatspilot.json', scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    
    try:
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])
        return pd.DataFrame(values[1:], columns=values[0]) if values else None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Ejemplo de uso
spreadsheet_id = '1igcRhzyqPU_Yp6lgb_bvhUuLNOJtd1NX5CZKpvLrDU4'  # Reemplaza con el ID de tu hoja de cálculo
range_name = 'AH!A1:Z1000'  # Reemplaza con el rango que deseas leer

df2 = read_from_sheets(spreadsheet_id, range_name)
df2 = df2.sort_values(by='Priority')
column_config = {
    "Register": st.column_config.LinkColumn(
        "Register",
        display_text="Register Data",
        help="Haz click para observar los registros"
    ),
    "Source": st.column_config.LinkColumn(
        "Source",
        display_text="Source Data",
        help="Haz click para descargar la información"
    )
}
st.dataframe(df2, column_config=column_config)
st.link_button("Source", "https://t.ly/PQm_h")
