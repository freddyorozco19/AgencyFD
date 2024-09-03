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
  PrioritySel = st.selectbox('Choose priority:', PriorityList)
  #df = df[df['Priority'] == PrioritySel].reset_index(drop=True)
st.dataframe(df)
