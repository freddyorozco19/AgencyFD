# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:14:41 2023
@author: Freddy J. Orozco R.
@Powered: WinStats.
"""

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
import io

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.header("M ANALYSIS")
st.divider()

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

# Configurar la columna de descarga como un hipervínculo
column_config = {
    "RD": st.column_config.LinkColumn(
        "RD",
        display_text="Register Data",
        help="Haz clic para descargar los datos de esta fila"
    )
}
# Muestra el DataFrame con la columna de hipervínculos
st.dataframe(df,
    column_config=column_config,
    hide_index=True,
    use_container_width=True)
