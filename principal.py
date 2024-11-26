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
#from PIL import Image
from matplotlib.patches import Rectangle
import math
import streamlit_antd_components as sac

############################################################################################################################################################################################################################
st.set_page_config(layout="wide")
st.logo("Resources/Img/FootballDivisionWorldwide2.png",icon_image="Resources/Img/FootballDivisionWorldwide2.png")
with st.sidebar:
    sac.menu([
        sac.MenuItem('home', icon='house-fill', tag=[sac.Tag('Tag1', color='green'), sac.Tag('Tag2', 'red')]),
        sac.MenuItem('products', icon='box-fill', children=[
            sac.MenuItem('apple', icon='apple'),
            sac.MenuItem('other', icon='git', description='other items', children=[
                sac.MenuItem('google', icon='google', description='item description'),
                sac.MenuItem('gitlab', icon='gitlab'),
                sac.MenuItem('wechat', icon='wechat'),
            ]),
        ]),
        sac.MenuItem('disabled', disabled=True),
        sac.MenuItem(type='divider'),
        sac.MenuItem('link', type='group', children=[
            sac.MenuItem('antd-menu', icon='heart-fill', href='cont/MatchAnalysis.py'),
            sac.MenuItem('bootstrap-icon', icon='bootstrap-fill', href='https://icons.getbootstrap.com/'),
        ]),
    ], open_all=True)
#nav.run()

#st.sidebar.link_button("Source", "https://t.ly/r78Av")

