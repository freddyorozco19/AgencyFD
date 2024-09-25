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

############################################################################################################################################################################################################################
def split_minute(minute_str):
    if '+' in minute_str:
        parts = minute_str.split('+')
        return parts[0], parts[1]
    else:
        return minute_str, ''

def calcular_radio_desde_gol(x, y, y_gol):
    radio = math.sqrt(x**2 + (y - y_gol)**2)
    return radio
def es_deep_completion(row):
    radio = math.sqrt(row['X1']**2 + (row['Y1'] - y_gol_oponente)**2)
    return radio <= radio_umbral
def esta_dentro_semicircunferencia(x, y):
    distancia_centro = math.sqrt((x - punto_medio_cancha_rival[0])**2 + (y - punto_medio_cancha_rival[1])**2)
    return distancia_centro <= radio_umbral

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_entre_puntos(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#def to_excel(df):
    #output = BytesIO()
    #writer = pd.ExcelWriter(output, engine='xlsxwriter')
    #df.to_excel(writer, index=False, sheet_name='Sheet1')
    #workbook = writer.book
    #worksheet = writer.sheets['Sheet1']
    #format1 = workbook.add_format({'num_format': '0.00'}) 
    #worksheet.set_column('A:A', None, format1)  
    #writer.save()
    #processed_data = output.getvalue()
    #return processed_data
def to_excelA(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def colorlist(color1, color2, num):
    """Generate list of num colors blending from color1 to color2"""
    result = [np.array(color1), np.array(color2)]
    while len(result) < num:
        temp = [result[0]]
        for i in range(len(result)-1):
            temp.append(np.sqrt((result[i]**2+result[i+1]**2)/2))
            temp.append(result[i+1])
        result = temp
    indices = np.linspace(0, len(result)-1, num).round().astype(int)
    return [result[i] for i in indices] 

hex_list2 = ['#121214', '#D81149', '#FF0050']
#hex_list = ['#121214', '#112F66', '#004DDD']B91845
hex_list4 = ['#5A9212', '#70BD0C', '#83E604']
#hex_list1 = ['#121214', '#854600', '#C36700']
hex_list = ['#121214', '#545454', '#9F9F9F']
hex_list1 = ['#121214', '#695E00', '#C7B200']
#hex_list2 = ['#121214', '#112F66', '#004DDD']
#hex_list = ['#121214', '#11834C', '#00D570']
cmap = sns.cubehelix_palette(start=.25, rot=-.3, light=1, reverse=True, as_cmap=True)
cmap2 = sns.diverging_palette(250, 344, as_cmap=True, center="dark")
cmap3 = sns.color_palette("dark:#FF0046", as_cmap=True)

def hex_to_rgb(value):
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

#####################################################################################################################################################

font_path = 'Resources/keymer-bold.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop2 = font_manager.FontProperties(fname=font_path)

font_path2 = 'Resources/BasierCircle-Italic.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path2)
prop3 = font_manager.FontProperties(fname=font_path2)

###########################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.header("PILOT DATA")

with st.form(key='form4'):
    formdata01, formdata02 = st.columns(2)
    with formdata01:
        uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    with formdata02:
        DataMode = st.checkbox("Activate comma separator")
    submit_button2 = st.form_submit_button(label='Aceptar')

if uploaded_file is not None:
    if DataMode:
        df = pd.read_csv(uploaded_file, sep=',')
    else:
        df = pd.read_csv(uploaded_file, sep=';')
else:
    df = pd.read_csv("Data/2022_4_Full.csv")

#Depuración data
df['season'] = df['season'].astype(str)
df['matchID'] = df['matchID'].astype(str)
df = df.rename(columns={'home_score': 'scorehome', 'away_score': 'scoreaway'})
MetricsHomeTotal = ["home_dfl_duels_accuracy_total", "home_dfl_duels_total", "home_dfl_duels_lost_total", "home_dfl_duels_won_total", "home_tackles_accuracy_total", "home_shots_accuracy_total", "home_crosses_accuracy_total", "home_passes_opponents_half_accuracy_total", "home_passes_accuracy_total", "home_duels_accuracy_total", "home_aerial_duels_accuracy_total", "home_tackles_won_total", "home_corners_won_total", "home_won_contest_total", "home_cards_yellow_total", "home_through_balls_total", "home_tackles_total", "home_shots_total", "home_cards_red_total", "home_passes_total", "home_offsides_total", "home_passes_long_total", "home_passes_opponents_half_total", "home_passes_final_third_total", "home_crosses_not_corners_total", "home_crosses_total", "home_clearances_total", "home_assists_total", "home_put_throughs_successful_total", "home_passes_final_third_successful_total", "home_fifty_fifties_successful_total", "home_shots_off_target_total", "home_passes_right_total", "home_put_throughs_total", "home_possession_total", "home_possession_won_middle_3rd_total", "home_possession_won_defensive_3rd_total", "home_possession_won_attacking_3rd_total", "home_possession_lost_all_total", "home_pen_area_entries_total", "home_shots_on_target_total", "home_passes_left_total", "home_interceptions_total", "home_goals_total", "home_passes_forward_total", "home_freekick_crosses_total", "home_formation_used_total", "home_fouls_won_total", "home_fouls_conceded_total", "home_final_third_entries_total", "home_fifty_fifties_total", "home_duels_won_total", "home_duels_lost_total", "home_corner_taken_total", "home_shots_blocked_total", "home_recoveries_total", "home_passes_backward_total", "home_shots_conceded_outside_box_total", "home_shots_conceded_inside_box_total", "home_own_goals_outside_box_total", "home_own_goals_inside_box_total", "home_att_hd_total_total", "home_aerial_duels_won_total", "home_aerial_duels_lost_total", "home_passes_successful_total", "home_passes_opponents_half_successful_total", "home_crosses_successful_total", "home_touches_total"]
MetricsAwayTotal = ["away_dfl_duels_accuracy_total", "away_dfl_duels_total", "away_dfl_duels_lost_total", "away_dfl_duels_won_total", "away_tackles_accuracy_total", "away_shots_accuracy_total", "away_crosses_accuracy_total", "away_passes_opponents_half_accuracy_total", "away_passes_accuracy_total", "away_duels_accuracy_total", "away_aerial_duels_accuracy_total", "away_tackles_won_total", "away_corners_won_total", "away_won_contest_total", "away_cards_yellow_total", "away_through_balls_total", "away_tackles_total", "away_shots_total", "away_cards_red_total", "away_passes_total", "away_offsides_total", "away_passes_long_total", "away_passes_opponents_half_total", "away_passes_final_third_total", "away_crosses_not_corners_total", "away_crosses_total", "away_clearances_total", "away_assists_total", "away_put_throughs_successful_total", "away_passes_final_third_successful_total", "away_fifty_fifties_successful_total", "away_shots_off_target_total", "away_passes_right_total", "away_put_throughs_total", "away_possession_total", "away_possession_won_middle_3rd_total", "away_possession_won_defensive_3rd_total", "away_possession_won_attacking_3rd_total", "away_possession_lost_all_total", "away_pen_area_entries_total", "away_shots_on_target_total", "away_passes_left_total", "away_interceptions_total", "away_goals_total", "away_passes_forward_total", "away_freekick_crosses_total", "away_formation_used_total", "away_fouls_won_total", "away_fouls_conceded_total", "away_final_third_entries_total", "away_fifty_fifties_total", "away_duels_won_total", "away_duels_lost_total", "away_corner_taken_total", "away_shots_blocked_total", "away_recoveries_total", "away_passes_backward_total", "away_shots_conceded_outside_box_total", "away_shots_conceded_inside_box_total", "away_own_goals_outside_box_total", "away_own_goals_inside_box_total", "away_att_hd_total_total", "away_aerial_duels_won_total", "away_aerial_duels_lost_total", "away_passes_successful_total", "away_passes_opponents_half_successful_total", "away_crosses_successful_total", "away_touches_total"]
df0 = df[['date', 'matchID', 'matchday', 'home', 'away', 'scorehome', 'scoreaway'] + MetricsHomeTotal + MetricsAwayTotal]
# Duplicar filas y crear la nueva columna "TeamSelName" con valores "Home" y "Away"
df0_home = df0.copy()
df0_away = df0.copy()
df0_home['TeamSelName'] = 'Home'
df0_away['TeamSelName'] = 'Away'
# Concatenar los dos dataframes (Home y Away) uno debajo del otro
dfT00 = pd.concat([df0_home, df0_away], ignore_index=True)
#dfT00 = dfT00[['TeamSelName'] + dfT00.columns.tolist()]
MetricsTotalConcat = MetricsHomeTotal + MetricsAwayTotal
#st.dataframe(dfT00)
df_list = []
# Iteramos sobre las filas del DataFrame original
for index, row in dfT00.iterrows():
    if row['TeamSelName'] == 'Home':
        # Seleccionamos las columnas que empiezan con 'home_' y eliminamos el prefijo
        new_row = row[dfT00.columns[dfT00.columns.str.startswith('home_')]].rename(lambda x: x.replace('home_', ''), axis=0)
    else:
        # Seleccionamos las columnas que empiezan con 'away_' y eliminamos el prefijo
        new_row = row[dfT00.columns[dfT00.columns.str.startswith('away_')]].rename(lambda x: x.replace('away_', ''), axis=0)
    # Añadimos la nueva fila al DataFrame de salida
    df_list.append(new_row)
# Creamos un nuevo DataFrame a partir de la lista de filas
new_df = pd.DataFrame(df_list).reset_index(drop=True)
new_df00 = pd.concat([dfT00['date'], dfT00['matchday'], dfT00['matchID'], dfT00['home'], dfT00['away'], dfT00['TeamSelName'], new_df], axis=1)
new_df00 = new_df00.sort_values(by='date', ascending=False)
# Asumiendo que tu DataFrame se llama 'df'
def select_team(row):
    if 'Home' in row['TeamSelName']:
        return row['home']
    elif 'Away' in row['TeamSelName']:
        return row['away']
    else:
        return None  # O puedes manejar otros casos según lo necesites

# Crear la nueva columna 'TeamSel'
new_df00.insert(5, 'SelName', new_df00.apply(select_team, axis=1))
#new_df00['TeamSel'] = new_df00.apply(select_team, axis=1)
st.dataframe(new_df00)
st.divider()

menuedt01, menuedt02, menuedt03, menuedt04, menuedt05 = st.columns(5)
with menuedt01:
    MatchdaySL = new_df00['matchday'].drop_duplicates().tolist()
    MatchdaySL.insert(0, "All")
    MatchdaySel = st.selectbox('Choose Matchday:', MatchdaySL)
    df_bk02 = new_df00
    if MatchdaySel == 'All':
        new_df00 = df_bk02
    else:
        new_df00 = new_df00[new_df00['matchday'] == MatchdaySel].reset_index(drop=True)

with menuedt02:
    TeamsSL = new_df00['SelName'].drop_duplicates().tolist()
    TeamsSL.insert(0, "All")
    TeamSel = st.selectbox('Choose Team:', TeamsSL)
    df_bk01 = new_df00
    if TeamSel == 'All':
        new_df00 = df_bk01
    else:
        new_df00 = new_df00[new_df00['SelName'] == TeamSel].reset_index(drop=True)
        
def to_numeric_safe(x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return np.nan
columns_to_process = new_df00.columns[6:]
#st.write(columns_to_process)
# Crear nuevas columnas con los percentiles solo para las columnas seleccionadas
#for column in columns_to_process:
    #new_column_name = f"{column}_PCN"
    #new_df00[new_column_name] = new_df00[column].rank(pct=True)

for column in columns_to_process:
    # Convertir la columna a numérico, manejando posibles errores
    numeric_column = new_df00[column].apply(to_numeric_safe)
    # Calcular el percentil solo para valores numéricos no nulos
    new_column_name = f"{column}_PCN"
    new_df00[new_column_name] = numeric_column.rank(pct=True, method='min')

st.dataframe(new_df00)
