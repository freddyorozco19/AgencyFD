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
############################################################################################################################################################################################################################

#Depuración data
df['season'] = df['season'].astype(str)
df['matchID'] = df['matchID'].astype(str)
df = df.rename(columns={'home_score': 'scorehome', 'away_score': 'scoreaway'})
MetricsHomeTotal = ["home_dfl_duels_accuracy_total", "home_dfl_duels_total", "home_dfl_duels_lost_total", "home_dfl_duels_won_total", "home_tackles_accuracy_total", "home_shots_accuracy_total", "home_crosses_accuracy_total", "home_passes_opponents_half_accuracy_total", "home_passes_accuracy_total", "home_duels_accuracy_total", "home_aerial_duels_accuracy_total", "home_tackles_won_total", "home_corners_won_total", "home_won_contest_total", "home_cards_yellow_total", "home_through_balls_total", "home_tackles_total", "home_shots_total", "home_cards_red_total", "home_passes_total", "home_offsides_total", "home_passes_long_total", "home_passes_opponents_half_total", "home_passes_final_third_total", "home_crosses_not_corners_total", "home_crosses_total", "home_clearances_total", "home_assists_total", "home_put_throughs_successful_total", "home_passes_final_third_successful_total", "home_fifty_fifties_successful_total", "home_shots_off_target_total", "home_passes_right_total", "home_put_throughs_total", "home_possession_total", "home_possession_won_middle_3rd_total", "home_possession_won_defensive_3rd_total", "home_possession_won_attacking_3rd_total", "home_possession_lost_all_total", "home_pen_area_entries_total", "home_shots_on_target_total", "home_passes_left_total", "home_interceptions_total", "home_goals_total", "home_passes_forward_total", "home_freekick_crosses_total", "home_formation_used_total", "home_fouls_won_total", "home_fouls_conceded_total", "home_final_third_entries_total", "home_fifty_fifties_total", "home_duels_won_total", "home_duels_lost_total", "home_corner_taken_total", "home_shots_blocked_total", "home_recoveries_total", "home_passes_backward_total", "home_shots_conceded_outside_box_total", "home_shots_conceded_inside_box_total", "home_own_goals_outside_box_total", "home_own_goals_inside_box_total", "home_att_hd_total_total", "home_aerial_duels_won_total", "home_aerial_duels_lost_total", "home_passes_successful_total", "home_passes_opponents_half_successful_total", "home_crosses_successful_total", "home_touches_total"]
MetricsAwayTotal = ["away_dfl_duels_accuracy_total", "away_dfl_duels_total", "away_dfl_duels_lost_total", "away_dfl_duels_won_total", "away_tackles_accuracy_total", "away_shots_accuracy_total", "away_crosses_accuracy_total", "away_passes_opponents_half_accuracy_total", "away_passes_accuracy_total", "away_duels_accuracy_total", "away_aerial_duels_accuracy_total", "away_tackles_won_total", "away_corners_won_total", "away_won_contest_total", "away_cards_yellow_total", "away_through_balls_total", "away_tackles_total", "away_shots_total", "away_cards_red_total", "away_passes_total", "away_offsides_total", "away_passes_long_total", "away_passes_opponents_half_total", "away_passes_final_third_total", "away_crosses_not_corners_total", "away_crosses_total", "away_clearances_total", "away_assists_total", "away_put_throughs_successful_total", "away_passes_final_third_successful_total", "away_fifty_fifties_successful_total", "away_shots_off_target_total", "away_passes_right_total", "away_put_throughs_total", "away_possession_total", "away_possession_won_middle_3rd_total", "away_possession_won_defensive_3rd_total", "away_possession_won_attacking_3rd_total", "away_possession_lost_all_total", "away_pen_area_entries_total", "away_shots_on_target_total", "away_passes_left_total", "away_interceptions_total", "away_goals_total", "away_passes_forward_total", "away_freekick_crosses_total", "away_formation_used_total", "away_fouls_won_total", "away_fouls_conceded_total", "away_final_third_entries_total", "away_fifty_fifties_total", "away_duels_won_total", "away_duels_lost_total", "away_corner_taken_total", "away_shots_blocked_total", "away_recoveries_total", "away_passes_backward_total", "away_shots_conceded_outside_box_total", "away_shots_conceded_inside_box_total", "away_own_goals_outside_box_total", "away_own_goals_inside_box_total", "away_att_hd_total_total", "away_aerial_duels_won_total", "away_aerial_duels_lost_total", "away_passes_successful_total", "away_passes_opponents_half_successful_total", "away_crosses_successful_total", "away_touches_total"]
MetricsTTotal = ["touches_total", "duels_accuracy_total", "duels_won_total", "duels_lost_total", "dfl_duels_total", "dfl_duels_accuracy_total", "dfl_duels_won_total", "dfl_duels_lost_total", "aerial_duels_accuracy_total", "aerial_duels_won_total", "aerial_duels_lost_total", "possession_total", "shots_total", "shots_accuracy_total", "shots_on_target_total", "shots_off_target_total", "shots_blocked_total", "goals_total", "assists_total", "passes_total", "passes_accuracy_total", "passes_successful_total", "passes_forward_total", "passes_right_total", "passes_left_total", "passes_backward_total", "passes_final_third_total", "passes_final_third_successful_total", "passes_long_total", "passes_opponents_half_total", "passes_opponents_half_accuracy_total", "passes_opponents_half_successful_total", "through_balls_total", "crosses_total", "crosses_accuracy_total", "crosses_successful_total", "crosses_not_corners_total", "final_third_entries_total", "pen_area_entries_total", "put_throughs_total", "put_throughs_successful_total", "tackles_total", "tackles_accuracy_total", "tackles_won_total", "clearances_total", "interceptions_total", "recoveries_total", "won_contest_total", "possession_won_attacking_3rd_total", "possession_won_middle_3rd_total", "possession_won_defensive_3rd_total", "shots_conceded_outside_box_total", "shots_conceded_inside_box_total", "fifty_fifties_total", "fifty_fifties_successful_total", "fouls_won_total", "fouls_conceded_total", "corners_won_total", "corner_taken_total", "freekick_crosses_total", "possession_lost_all_total", "att_hd_total_total", "own_goals_outside_box_total", "own_goals_inside_box_total", "cards_yellow_total", "cards_red_total", "offsides_total"]
df0 = df[['date', 'matchID', 'matchday', 'home', 'away', 'scorehome', 'scoreaway'] + MetricsHomeTotal + MetricsAwayTotal]
############################################################################################################################################################################################################################

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
new_df00 = new_df00[['date', 'matchday', 'matchID', 'home', 'away', 'SelName', 'TeamSelName'] + MetricsTTotal]
new_df00bk = new_df00

#new_df00['TeamSel'] = new_df00.apply(select_team, axis=1)
st.dataframe(new_df00)
st.divider()
############################################################################################################################################################################################################################

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
with menuedt03:
    SelTeamSL = ['All', 'Home', 'Away']
    SelTeamSel = st.selectbox('Choose Type:', SelTeamSL)
    df_bk03 = new_df00
    if SelTeamSel == 'All':
        new_df00 = df_bk03
    else:
        new_df00 = new_df00[new_df00['TeamSelName'] == SelTeamSel].reset_index(drop=True)

############################################################################################################################################################################################################################

def to_numeric_safe(x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return np.nan
columns_to_process = new_df00.columns[7:]


nuevas_columnas = []

for column in columns_to_process:
    # Convertir la columna a numérico, manejando posibles errores
    numeric_column = new_df00[column].apply(to_numeric_safe)
    # Calcular el percentil solo para valores numéricos no nulos
    new_column_name = f"{column}_PCN"
    new_df00[new_column_name] = numeric_column.rank(pct=True, method='min')
    nuevas_columnas.append(new_column_name)

#st.dataframe(new_df00)
#st.write(nuevas_columnas)
new_df10 = new_df00[['matchID', 'SelName'] + nuevas_columnas]
#st.dataframe(new_df00[['matchID'] + nuevas_columnas])
############################################################################################################################################################################################################################


def generate_progress_column_config(columns_list):
    column_config = {}
    for col in columns_list:
        # Extraer el nombre corto de la columna para el label
        short_name = col.replace('_PCN_total', '')
        column_config[col] = st.column_config.ProgressColumn(
            label=short_name,
            help=col,
            format="%.2f",
            min_value=0,
            max_value=1)
    return column_config

column_config = generate_progress_column_config(nuevas_columnas)
st.dataframe(new_df10, column_config = column_config)
st.divider()
############################################################################################################################################################################################################################
df = new_df00bk
df = df.sort_values(by='touches_total', ascending=False).reset_index(drop=True)
MatchIDList = df['matchID'].drop_duplicates().tolist()
MatchIDSel = st.selectbox('Choose MatchID:', MatchIDList)
scaler =  StandardScaler()
#scaled_values1 = scaler.fit_transform(df['touches_total'].values.reshape(-1, 1))
scaled_values = scaler.fit_transform(df[MetricsTTotal])
dfscaled = pd.DataFrame(scaled_values, columns=MetricsTTotal)
dfscaledC = pd.concat([df['matchID'], dfscaled], axis=1)
#st.write(np.mean(new_df00bk['touches_total']))
#st.write(dfscaledC)
#st.write(new_df00bk)
############################################################################################################################################################################################################################

fig, ax = plt.subplots(figsize=(3, 10), dpi = 300)
ax.axis("off")
ax.set_xlim(-4, 4)
ax.set_ylim(1, 70)
fig.patch.set_visible(False)

#Spines
spines = ["top", "right", "bottom", "left"]
for s in spines:
    ax.spines[s].set_visible(False)
    
ax.text(-7.5, 67.75, 'TOUCHES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 66.75, 'DUELS (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 65.75, 'DUELS WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 64.75, 'DUELS LOST',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 63.75, 'DFL DUELS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 62.75, 'DFL DUELS (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 61.75, 'DFL DUELS WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 60.75, 'DFL DUELS LOST',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 59.75, 'AERIAL DUELS (%)',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 58.75, 'AERIAL DUELS WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 57.75, 'AERIAL DUELS LOST',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 56.75, 'POSSESSION',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 55.75, 'SHOTS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 54.75, 'SHOTS (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 53.75, 'SHOTS ON TARGET',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 52.75, 'SHOTS OFF TARGET',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 51.75, 'SHOTS BLOCKED',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 50.75, 'GOALS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 49.75, 'ASSISTS',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 48.75, 'PASSES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 47.75, 'PASSES (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 46.75, 'PASSES SUCCESSFUL',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 45.75, 'PASSES FORWARD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 44.75, 'PASSES RIGHT',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 43.75, 'PASSES LEFT',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 42.75, 'PASSES BACKWARD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 41.75, 'PASSES FINAL THIRD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 40.75, 'PASSES SUCC. FINAL THIRD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 39.75, 'PASSES LONG',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 38.75, 'PASSES OPPONENTS HALF',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 37.75, 'PASSES OPPONENTS HALF (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 36.75, 'PASSES SUCC. OPPONENTS HALF',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 35.75, 'THROUGH BALLS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 34.75, 'CROSSES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 33.75, 'CROSSES (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 32.75, 'CROSSES SUCCESSFUL',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 31.75, 'CROSSES NOT CORNERS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 30.75, 'FINAL THIRD ENTRIES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 29.75, 'PENALTY AREA ENTRIES',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 28.75, 'PUT THROUGHS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 27.75, 'PUT THROUGHS SUCCESSFUL',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 26.75, 'TACKLES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 25.75, 'TACKLES (%)',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 24.75, 'TACKLES WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 23.75, 'CLEARANCES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 22.75, 'INTERCEPTIONS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 21.75, 'RECOVERIES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 20.75, 'WON CONTEST',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 19.75, 'POSSESSION WON ATT-3RD',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 18.75, 'POSSESSION WON MID-3RD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 17.75, 'POSSESSION WON DEF-3RD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 16.75, 'SHOTS CONCEDED OUTSIDE BOX',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 15.75, 'SHOTS CONCEDED INSIDE BOX',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 14.75, 'FIFTY FIFTIES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 13.75, 'FIFTY FIFTIES SUCCESSFUL',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 12.75, 'FOULS WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 11.75, 'FOULS CONCEDED',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 10.75, 'CORNERS WON',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 9.75, 'CORNERS TAKEN',color='w', fontproperties=prop2, fontsize=5)

ax.text(-7.5, 8.75, 'FREEKICK CROSSES',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 7.75, 'POSSESSION LOST ALL',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 6.75, 'ATT HD',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 5.75, 'OWN-GOALS OUTSIDE BOX',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 4.75, 'OWN-GOALS INSIDE BOX',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 3.75, 'YELLOW CARDS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 2.75, 'RED CARDS',color='w', fontproperties=prop2, fontsize=5)
ax.text(-7.5, 1.75, 'OFFSIDES',color='w', fontproperties=prop2, fontsize=5)

for i in range(2, 69):  # Desde 2 hasta 68, inclusive
    ax.plot([-4, 4], [i, i], color=(0.403921568627451, 0.403921568627451, 0.4431372549019608, 0.25), zorder=2, lw=7)

ax.plot([0, 0],[1, 69], color="w", ls="--", zorder=10, lw=0.5)

# Graficar todas las columnas seleccionadas
MetricsTTotal_invertido = MetricsTTotal[::-1]
for i, col in zip(range(2, 69), MetricsTTotal_invertido):  # Recorrer desde 2 hasta 68
    ax.scatter(dfscaledC[col], np.full(len(dfscaledC), i),  # Usar i directamente
               color="#FF0046", 
               s=30, 
               alpha=0.50, 
               linewidth=0.5, 
               zorder=3)


dfscaledC2 = dfscaledC[dfscaledC['matchID'] == MatchIDSel].reset_index(drop=True)
ax.scatter(dfscaledC2['touches_total'], [68]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['duels_accuracy_total'], [67]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['duels_won_total'], [66]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['duels_lost_total'], [65]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['dfl_duels_total'], [64]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['dfl_duels_accuracy_total'], [63]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['dfl_duels_won_total'], [62]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['dfl_duels_lost_total'], [61]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['aerial_duels_accuracy_total'], [60]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['aerial_duels_won_total'], [59]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['aerial_duels_lost_total'], [58]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['possession_total'], [57]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_total'], [56]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_accuracy_total'], [55]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_on_target_total'], [54]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_off_target_total'], [53]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_blocked_total'], [52]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['goals_total'], [51]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['assists_total'], [50]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['passes_total'], [49]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_accuracy_total'], [48]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_successful_total'], [47]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_forward_total'], [46]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_right_total'], [45]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_left_total'], [44]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_backward_total'], [43]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_final_third_total'], [42]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_final_third_successful_total'], [41]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_long_total'], [40]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['passes_opponents_half_total'], [39]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_opponents_half_accuracy_total'], [38]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['passes_opponents_half_successful_total'], [37]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['through_balls_total'], [36]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['crosses_total'], [35]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['crosses_accuracy_total'], [34]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['crosses_successful_total'], [33]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['crosses_not_corners_total'], [32]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['final_third_entries_total'], [31]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['pen_area_entries_total'], [30]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['put_throughs_total'], [29]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['put_throughs_successful_total'], [28]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['tackles_total'], [27]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['tackles_accuracy_total'], [26]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['tackles_won_total'], [25]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['clearances_total'], [24]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['interceptions_total'], [23]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['recoveries_total'], [22]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['won_contest_total'], [21]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['possession_won_attacking_3rd_total'], [20]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['possession_won_middle_3rd_total'], [19]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['possession_won_defensive_3rd_total'], [18]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_conceded_outside_box_total'], [17]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['shots_conceded_inside_box_total'], [16]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['fifty_fifties_total'], [15]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['fifty_fifties_successful_total'], [14]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['fouls_won_total'], [13]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['fouls_conceded_total'], [12]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['corners_won_total'], [11]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['corner_taken_total'], [10]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

ax.scatter(dfscaledC2['freekick_crosses_total'], [9]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['possession_lost_all_total'], [8]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['att_hd_total_total'], [7]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['own_goals_outside_box_total'], [6]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['own_goals_inside_box_total'], [5]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['cards_yellow_total'], [4]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['cards_red_total'], [3]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)
ax.scatter(dfscaledC2['offsides_total'], [2]*len(dfscaledC2), edgecolor="#121214", color="w", s=40, alpha=0.75, linewidth=0.5, zorder=3)

st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
st.divider()
