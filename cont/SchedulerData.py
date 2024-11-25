import streamlit as st
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
import altair as alt
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.header("SCHEDULER DATA")
st.divider()

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
spreadsheet_id = '1SZx0fB0BZQJruQ-usmVNu43rMP2z7RNKsP3FhEdI9Uk'  # Reemplaza con el ID de tu hoja de cálculo
range_name = 'Hoja 3!A1:Z1000'  # Reemplaza con el rango que deseas leer

df = read_from_sheets(spreadsheet_id, range_name)
st.dataframe(df)
