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

############################################################################################################################################################################################################################

font_path = 'Resources/keymer-bold.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop2 = font_manager.FontProperties(fname=font_path)

font_path2 = 'Resources/BasierCircle-Italic.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path2)
prop3 = font_manager.FontProperties(fname=font_path2)

############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.header("EVENTING DATA")
st.divider()

############################################################################################################################################################################################################################

spreadsheet_id = '19im-aUSbVMeLgCgIGnsKPOvyJDiKbaE5SMIC_sDd4v4'  # Reemplaza con el ID de tu hoja de cálculo
range_name = 'Hoja 1!A1:Z1000'  # Reemplaza con el rango que deseas leer

df = read_from_sheets(spreadsheet_id, range_name)
column_config = {
    "Register": st.column_config.LinkColumn(
        "Register",
        display_text="Register Data",
        help="Haz click para observar los registros"),
    "Source": st.column_config.LinkColumn(
        "Source",
        display_text="Source Data",
        help="Haz click para descargar la información")
}
st.dataframe(df, column_config=column_config)

menuopt01, menuopt02, menuopt03 = st.columns(3)
with menuopt01:
    PlayersFDList = df['Player'].drop_duplicates().tolist()
    PlayersFDSel = st.selectbox("Choose Player", PlayersFDList)
with menuopt02:
    st.selectbox("Choose Competition", ['Premier League', 'Bundesliga', 'CONMEBOL Eliminatorias'])
with menuopt03:
    MatchIDFDList = df['MatchID'].drop_duplicates().tolist()
    MatchIDFDSel = st.selectbox("Choose MatchID", MatchIDFDList)


with st.container(border=True):
    menuoptcon01, menuoptcon02, menuoptcon03, menuoptcon04 = st.columns(4)
    with menuoptcon01:
        PlotVizSelFDData = st.selectbox("Choose Metric", ['Actions', 'Passes', 'Shots', 'Def. Actions', 'Possession'])
   
#container = st.container(border=True)
#container.write("This is inside the container")
#st.write("This is outside the container")
# Now insert some more in the container
#container.write("This is inside too")
    df['EfectiveMinute'] = df['EfectiveMinute'].astype(int)
    df['X1'] = df['X1'].str.replace(',','.').astype(float)
    df['Y1'] = df['Y1'].str.replace(',','.').astype(float)
    df['X2'] = df['X2'].str.replace(',','.').astype(float)
    df['Y2'] = df['Y2'].str.replace(',','.').astype(float)

    MaxAddMin = df['EfectiveMinute'].max()
    if PlotVizSelFDData == "Actions":
        with menuoptcon02:
            OptionPlot = ['Touches Map', 'Touches Zones - Heatmap', 'Touches Gaussian - Heatmap', 'Touches Kernel - Heatmap', 'Territory Actions Map', 'Touches Opponent Field Map', 'Touches Opponent Field - Heatmap', 'Touches Final Third', 'Touches Final Third - Heatmap', 'Touches Penalty Area']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        with menuoptcon03:
            EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
        if OptionPlotSel == 'Territory Actions Map': 
            with menuoptcon04:
                ColorOptionSel = st.color_picker('Selecciona color:', '#FF0046')
                colorviz = ColorOptionSel
        else:
            with menuoptcon04:
                SelOpt = ['WinStats', 'FD']
                ColorOptionSel = st.selectbox('Selecciona color:', SelOpt)
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            if (OptionPlotSel == 'Touches Opponent Field Map') | (OptionPlotSel == 'Touches Opponent Field - Heatmap') | (OptionPlotSel == 'Touches Final Third') | (OptionPlotSel == 'Touches Final Third - Heatmap') | (OptionPlotSel == 'Touches Penalty Area'):
                pitch = VerticalPitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68, half=True)
            else:
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
                #Adding directon arrow
                ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
                ax29.axis("off")
                ax29.set_xlim(0,10)
                ax29.set_ylim(0,10)
                ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            pitch.draw(ax=ax)
            
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            img_path = "Resources/Img/winstats_pplwhite.png"
            img = Image.open(img_path)
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")

            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            if ColorOptionSel == 'WinStats':
                hex_list2 = ['#121214', '#D81149', '#FF0050']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#FF0050"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
                green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
                blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            if ColorOptionSel == 'FD':
                hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#83E604"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
                green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
                blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            #df = dfKK.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
            #st.write(df)
            if OptionPlotSel == 'Touches Map': 
                    
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 4.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 0, 'ACCIONES\nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

            if OptionPlotSel == 'Touches Opponent Field Map':
            
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                df = df[df['Y1'] >= 52.5].reset_index()
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(34, 108, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES EN CAMPO RIVAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.5, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Touches Opponent Field - Heatmap':
    
                df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                df = df[df['Y1'] >= 52.5].reset_index()
                zone_areas = {
                    'zone_1': {'x_lower_bound': 54.16, 'x_upper_bound': 68, 'y_lower_bound': 88.5, 'y_upper_bound': 105},
                    'zone_2': {'x_lower_bound': 0, 'x_upper_bound': 13.84, 'y_lower_bound': 88.5, 'y_upper_bound': 105},
                    'zone_3': {'x_lower_bound': 54.16, 'x_upper_bound': 68, 'y_lower_bound': 70.5, 'y_upper_bound': 88.5},
                    'zone_4': {'x_lower_bound': 0, 'x_upper_bound': 13.84, 'y_lower_bound': 70.5, 'y_upper_bound': 88.5},
                    'zone_5': {'x_lower_bound': 43.16, 'x_upper_bound': 54.16, 'y_lower_bound': 88.5, 'y_upper_bound': 105},
                    'zone_6': {'x_lower_bound': 13.84, 'x_upper_bound': 24.84, 'y_lower_bound': 88.5, 'y_upper_bound': 105},
                    'zone_7': {'x_lower_bound': 24.84, 'x_upper_bound': 43.16, 'y_lower_bound': 88.5, 'y_upper_bound': 105},
                    'zone_8': {'x_lower_bound': 24.84, 'x_upper_bound': 43.16, 'y_lower_bound': 70.5, 'y_upper_bound': 88.5},
                    'zone_9': {'x_lower_bound': 43.16, 'x_upper_bound': 54.16, 'y_lower_bound': 70.5, 'y_upper_bound': 88.5},
                    'zone_10': {'x_lower_bound': 13.84, 'x_upper_bound': 24.84, 'y_lower_bound': 70.5, 'y_upper_bound': 88.5},
                    'zone_11': {'x_lower_bound': 43.16, 'x_upper_bound': 54.16, 'y_lower_bound': 52.5, 'y_upper_bound': 70.5},
                    'zone_12': {'x_lower_bound': 13.84, 'x_upper_bound': 24.84, 'y_lower_bound': 52.5, 'y_upper_bound': 70.5},
                    'zone_13': {'x_lower_bound': 54.16, 'x_upper_bound': 68, 'y_lower_bound': 52.5, 'y_upper_bound': 70.5},
                    'zone_14': {'x_lower_bound': 0, 'x_upper_bound': 13.84, 'y_lower_bound': 52.5, 'y_upper_bound': 70.5},
                    'zone_15': {'x_lower_bound': 24.84, 'x_upper_bound': 43.16, 'y_lower_bound': 52.5, 'y_upper_bound': 70.5}
                }
                
                def assign_action_zone(x,y):
                    '''
                    This function returns the zone based on the x & y coordinates of the shot
                    taken.
                    Args:
                        - x (float): the x position of the shot based on a vertical grid.
                        - y (float): the y position of the shot based on a vertical grid.
                    '''
                    global zone_areas
                    # Conditions
                    for zone in zone_areas:
                        if (x >= zone_areas[zone]['x_lower_bound']) & (x <= zone_areas[zone]['x_upper_bound']):
                            if (y >= zone_areas[zone]['y_lower_bound']) & (y <= zone_areas[zone]['y_upper_bound']):
                                return zone
                
                zone_colors = {
                    'zone_1': 'black',
                    'zone_2': 'red',
                    'zone_3': 'blue',
                    'zone_4': 'yellow',
                    'zone_5': 'green',
                    'zone_6': 'pink',
                    'zone_7': 'purple',
                    'zone_8': 'grey',
                    'zone_9': 'brown',
                    'zone_10': 'lightblue',
                    'zone_11': 'lightcyan',
                    'zone_12': 'lightgrey',
                    'zone_13': 'w',
                    'zone_14': 'orange',
                    'zone_15': 'cyan'
                }
                
                #df = df[df['y'] >= 52.5].reset_index()
                df['zone_area'] = [assign_action_zone(x,y) for x,y in zip(df['X1'], df['Y1'])]
                data = df.groupby(['zone_area']).apply(lambda x: x.shape[0]).reset_index()
                data.rename(columns={0:'num_actions'}, inplace=True)
                data['pct_actions'] = data['num_actions']/df['Event'].count()
                # Asegurar que todas las zonas estén representadas
                all_zones = pd.DataFrame({'zone_area': zone_areas.keys()})
                plot_df = all_zones.merge(data, on='zone_area', how='left').fillna(0)
                max_value = plot_df['pct_actions'].max()
                for zone in zone_areas:
                    action_pct = plot_df[plot_df['zone_area'] == zone]['pct_actions'].iloc[0] if zone in plot_df['zone_area'].values else 0
                    x_lim = [zone_areas[zone]['x_lower_bound'], zone_areas[zone]['x_upper_bound']]
                    y1 = zone_areas[zone]['y_lower_bound']
                    y2 = zone_areas[zone]['y_upper_bound']
                    # Si el porcentaje es 0, no dibujamos el fondo de color
                    if action_pct > 0:
                        ax.fill_between(
                            x=x_lim, 
                            y1=y1, y2=y2, 
                            color=colorviz, alpha=(action_pct/max_value),
                            zorder=0, ec='None')
                    x_pos = x_lim[0] + abs(x_lim[0] - x_lim[1])/2
                    y_pos = y1 + abs(y1 - y2)/2
                    text_ = ax.annotate(
                        xy=(x_pos, y_pos),
                        text=f'{action_pct:.0%}',
                        ha='center',
                        va='center',
                        color='w',
                        fontproperties=prop2,
                        size=20
                    )
                    text_.set_path_effects(
                        [path_effects.Stroke(linewidth=1.0, foreground='k'), path_effects.Normal()]
                    )
                ax.plot([13.84, 13.84], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([54.16, 54.16], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([24.84, 24.84], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([43.16, 43.16], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([0, 68], [88.5, 88.5], ls='--', color='#9F9F9F')
                ax.plot([0, 68], [70.5, 70.5], ls='--', color='#9F9F9F')
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.5, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax.text(34, 108, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES EN CAMPO RIVAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            if OptionPlotSel == 'Territory Actions Map': 
    
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
    
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                scaler  = StandardScaler()
                defpoints1 = df[['X1', 'Y1']].values
                defpoints2 = scaler.fit_transform(defpoints1)
                df2 = pd.DataFrame(defpoints2, columns = ['Xstd', 'Ystd'])
                df3 = pd.concat([df, df2], axis=1)
                df5=df3
                df3 = df3[df3['Xstd'] <= 1]
                df3 = df3[df3['Xstd'] >= -1]
                df3 = df3[df3['Ystd'] <= 1]
                df3 = df3[df3['Ystd'] >= -1].reset_index()
                df9 = df
                df = df3
                defpoints = df[['X1', 'Y1']].values
                #st.write(defpoints)
                hull = ConvexHull(df[['X1','Y1']])        
                ax.scatter(df9['X1'], df9['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                #Loop through each of the hull's simplices
                for simplex in hull.simplices:
                    #Draw a black line between each
                    ax.plot(defpoints[simplex, 0], defpoints[simplex, 1], '#BABABA', lw=2, zorder = 1, ls='--')
                ax.fill(defpoints[hull.vertices,0], defpoints[hull.vertices,1], colorviz, alpha=0.7)
                meanposx = df9['X1'].mean()
                meanposy = df9['Y1'].mean()
                ax.scatter(meanposx, meanposy, s=1000, color="w", edgecolors=colorviz, lw=2.5, zorder=25, alpha=0.95)
                names = PlayersFDSel.split()
                iniciales = ""
                for name in names:
                   iniciales += name[0] 
                #names_iniciales = names_iniciales.squeeze().tolist()
                ax.text(meanposx, meanposy, iniciales, color='k', fontproperties=prop2, fontsize=13, zorder=34, ha='center', va='center')
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -0.5, 'ACCIONES \nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            elif OptionPlotSel == 'Touches Zones - Heatmap':
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
    
                path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            elif OptionPlotSel == 'Touches Gaussian - Heatmap':
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
    
                bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            elif OptionPlotSel == 'Touches Kernel - Heatmap':
                
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                #bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                #bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                kde = pitch.kdeplot(dfKKcleaned.X1, dfKKcleaned.Y1, ax=ax,
                    # fill using 100 levels so it looks smooth
                    fill=True, levels=500,
                    # shade the lowest area so it looks smooth
                    # so even if there are no events it gets some color
                    thresh=0,
                    cut=2, alpha=0.7, zorder=-2,  # extended the cut so it reaches the bottom edge
                    cmap=cmaps)
    
                #pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Touches Final Third':
    
                df.rename(columns={'X1':'Y1', 'Y1':'X1', 'X2':'Y2', 'Y2':'X2'}, inplace=True)            
                df = df[df['Y1'] >= 70].reset_index()
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.hlines(y=70, xmin=0, xmax=68, color='w', alpha=0.3, ls='--', zorder=-1)
                #ax.add_patch(Rectangle((70, 0), 35, 68, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.add_patch(Rectangle((0, 70), 68, 35, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.text(34, 108, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES EN TERCIO FINAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.5, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Touches Final Third - Heatmap':
                
                ax.plot([13.84, 13.84], [70, 105], ls='--', color='w')
                ax.plot([54.16, 54.16], [70, 105], ls='--', color='w')
                ax.plot([24.84, 24.84], [70, 105], ls='--', color='w')
                ax.plot([43.16, 43.16], [70, 105], ls='--', color='w')
                ax.plot([0, 68], [70, 70], ls='--', color='w')
        
                zone_areas = {
                    'zone_1': {
                        'x_lower_bound': 54.16, 'x_upper_bound': 68,
                        'y_lower_bound': 70, 'y_upper_bound': 105,
                    },
                    'zone_2': {
                        'x_lower_bound': 0, 'x_upper_bound': 13.84,
                        'y_lower_bound': 70, 'y_upper_bound': 105,
                    },
                    'zone_3': {
                        'x_lower_bound': 43.16, 'x_upper_bound': 54.16,
                        'y_lower_bound': 70, 'y_upper_bound': 105,
                    },
                    'zone_4': {
                        'x_lower_bound': 13.84, 'x_upper_bound': 24.84,
                        'y_lower_bound': 70, 'y_upper_bound': 105,
                    },
                    'zone_5': {
                        'x_lower_bound': 24.84, 'x_upper_bound': 43.16,
                        'y_lower_bound': 70, 'y_upper_bound': 105,
                    }
                }
                
                def assign_action_zone(x,y):
                    '''
                    This function returns the zone based on the x & y coordinates of the shot
                    taken.
                    Args:
                        - x (float): the x position of the shot based on a vertical grid.
                        - y (float): the y position of the shot based on a vertical grid.
                    '''
                
                    global zone_areas
                
                    # Conditions
                
                    for zone in zone_areas:
                        if (x >= zone_areas[zone]['x_lower_bound']) & (x <= zone_areas[zone]['x_upper_bound']):
                            if (y >= zone_areas[zone]['y_lower_bound']) & (y <= zone_areas[zone]['y_upper_bound']):
                                return zone
                
                df.rename(columns={'X1':'Y1', 'Y1':'X1', 'X2':'Y2', 'Y2':'X2'}, inplace=True)            
                df = df[df['Y1'] >= 70].reset_index()
                df['zone_area'] = [assign_action_zone(x,y) for x,y in zip(df['X1'], df['Y1'])]
                
                # Asegurar que todas las zonas estén representadas
                all_zones = pd.DataFrame({'zone_area': zone_areas.keys()})
                data1 = df.groupby(['zone_area']).size().reset_index(name='num_actions')
                data1['pct_actions'] = data1['num_actions'] / df['Event'].count()
                plot_df1 = all_zones.merge(data1, on='zone_area', how='left').fillna(0)
                
                max_value1 = plot_df1['pct_actions'].max()
                
                for zone in zone_areas:
                    action_pct = plot_df1[plot_df1['zone_area'] == zone]['pct_actions'].iloc[0]
                    x_lim = [zone_areas[zone]['x_lower_bound'], zone_areas[zone]['x_upper_bound']]
                    y1 = zone_areas[zone]['y_lower_bound']
                    y2 = zone_areas[zone]['y_upper_bound']
                    
                    if action_pct > 0:
                        ax.fill_between(
                            x=x_lim, 
                            y1=y1, y2=y2, 
                            color='#F00040', alpha=(action_pct/max_value1),
                            zorder=0, ec='None')
                    
                    x_pos = x_lim[0] + abs(x_lim[0] - x_lim[1])/2
                    y_pos = y1 + abs(y1 - y2)/2
                    text_ = ax.annotate(
                        xy=(x_pos, y_pos),
                        text=f'{action_pct:.0%}',
                        ha='center',
                        va='center',
                        color='w',
                        fontproperties=prop2,
                        size=25
                    )
                    text_.set_path_effects(
                        [path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]
                    )
                ax.text(34, 108, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES EN TERCIO FINAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.5, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Touches Penalty Area':
    
                df.rename(columns={'X1':'Y1', 'Y1':'X1', 'X2':'Y2', 'Y2':'X2'}, inplace=True)            
                # Coordenadas del cuadrilátero
                #x1_cuadrilatero, y1_cuadrilatero = 88.5, 13.84
                x1_cuadrilatero, y1_cuadrilatero = 13.84, 88.5
                #x2_cuadrilatero, y2_cuadrilatero = 105, 13.84
                x2_cuadrilatero, y2_cuadrilatero = 13.84, 105
                #x3_cuadrilatero, y3_cuadrilatero = 88.5, 54.16
                x3_cuadrilatero, y3_cuadrilatero = 54.16, 88.5
                #x4_cuadrilatero, y4_cuadrilatero = 105, 54.16
                x4_cuadrilatero, y4_cuadrilatero = 54.16, 105
                condicion = (
                    (df['Y1'] >= y1_cuadrilatero) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['X1'] >= x1_cuadrilatero) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['Y1'] <= y4_cuadrilatero) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['X1'] <= x3_cuadrilatero)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
                
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[condicion]
                
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(34, 108, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " TOQUES EN ÁREA RIVAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax.set_ylim(52.3,110)
                #ax.vlines(x=88.5, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.vlines(x=105, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.hlines(xmin=88.5, xmax=105, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.hlines(xmin=88.5, xmax=105, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.5, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    if PlotVizSelFDData == "Shots":
        with menuoptcon02:
            OptionPlot = ['Shots Map', 'Shots Heatmap - Zones', 'Shots Heatmap - Gaussian']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        if OptionPlotSel == 'Shots Map': 
            with menuoptcon03:
                ColorOptionSel = st.color_picker('Selecciona color:', '#FF0046')
                colorviz = ColorOptionSel
        else:
            with menuoptcon03:
                SelOpt = ['WinStats', 'FD']
                ColorOptionSel = st.selectbox('Selecciona color:', SelOpt)

        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
    
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            if (OptionPlotSel == 'Shots Heatmap - Gaussian') | (OptionPlotSel == 'Shots Heatmap - Zones'):
                pitch = VerticalPitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
            else:
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
                #Adding directon arrow
                ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
                ax29.axis("off")
                ax29.set_xlim(0,10)
                ax29.set_ylim(0,10)
                ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            pitch.draw(ax=ax)
            
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            img_path = "Resources/Img/winstats_pplwhite.png"
            img = Image.open(img_path)
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            #df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
    
            df = df[(df['Event'] == 'Goals') | (df['Event'] == 'Shots on target') | (df['Event'] == 'Shots off target') | (df['Event'] == 'Blocks')].reset_index(drop=True)
            #df = df[df['MatchID'].isin(ListMatchSel)]
            dfKK = df
            if ColorOptionSel == 'WinStats':
                hex_list2 = ['#121214', '#D81149', '#FF0050']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#FF0050"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
                green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
                blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            if ColorOptionSel == 'FD':
                hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#83E604"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
                green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
                blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)



            if OptionPlotSel == 'Shots Map': 
                dfKK = df
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKK)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 2.25, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -1.25, 'REMATES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                #ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                #ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == 'Shots Heatmap - Gaussian':
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                ax.scatter(df['X1'], df['Y1'], color = "#A4BACF", edgecolors="#FFF", alpha=0.5, s=100, zorder=10)
                kdeplot = pitch.kdeplot(df.Y1, df.X1, ax=ax, cmap=get_continuous_cmap(hex_list), fill=True, levels=100, zorder=-1, alpha=0.5)
                ax.set_ylim(52.3,110)
                #ax.text(52.5,70, "" + PlayerSelExpData_txt.upper() + " - " + str(len(dfKK)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding colorbar
                ax9 = fig.add_axes([0.14,0.13,0.20,0.07])
                ax9.scatter(5.75,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=1.0)
                ax9.scatter(4.00,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=0.7)
                ax9.scatter(2.25,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=0.2)
                ax9.text(4, 0.25, '-  REMATES  +',c='w', fontproperties=prop2, fontsize=10, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
        
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Shots Heatmap - Zones':
                def soc_pitch_divisions(ax, grids = False):
                    '''
                    This function returns a vertical football pitch
                    divided in specific locations.
                
                    Args:
                        ax (obj): a matplotlib axes.
                        grids (bool): should we draw the grid lines?
                    '''
    
                    pitch.draw(ax = ax)
                
                    # Where we'll draw the lines
                    if grids:
                        y_lines = [105 - 5.5*x for x in range(1,10)]
                        x_lines = [68 - 6.8*x for x in range(1,10)]
                
                        for i in x_lines:
                            ax.plot(
                                [i, i], [45, 105], 
                                color = "#8E8E8E", 
                                ls = "--",
                                lw = 0.3,
                                zorder = -1)
                        for j in y_lines:
                            ax.plot(
                                [68, 0], [j, j],
                                color = "#8E8E8E", 
                                ls = "--",
                                lw = 0.3,
                                zorder = -1)
                
                    return ax
                #df = df[df['e'] == 'Shot'].reset_index(drop=True)
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
        
                soc_pitch_divisions(ax, grids = True)
        
                #df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                
                # We define the cuts for our data (same as our pitch divisions)
                # Only difference is we need to add the edges
                
                y_bins = [105] + [105 - 5.5*x for x in range(1,10)] + [45]
                x_bins = [68] + [68 - 6.8*x for x in range(1,10)] + [0]
                
                x_bins.sort()
                y_bins.sort()
        
                df["bins_x"] = pd.cut(df["X1"], bins = x_bins)            
                df["bins_y"] = pd.cut(df["Y1"], bins = y_bins)
        
                #Group and sum xGOT by side and location
                df_teams = (df.groupby(["bins_x", "bins_y"], observed = True)["Event"].count().reset_index()) 
                # And we sort it based on the bins_y and bins_x columns
                df_teams = (df_teams.sort_values(by = ["bins_y", "bins_x"]).reset_index(drop = True))
                
                example_df = df_teams
                total_example = example_df["Event"].sum()
                
                # Compute share of xGOT as a % of total
                example_df = (example_df.assign(COUNT_share = lambda x: x.Event/total_example))
                # Scale data to the maximum value to get a nice color scale
                example_df = (example_df.assign(COUNT_scaled = lambda x: x.COUNT_share/x.COUNT_share.max()))
                counter = 0
                for X, Y in zip(example_df["bins_x"], example_df["bins_y"]):
                    #This colours our bins
                    ax.fill_between(
                        x = [X.left, X.right],
                        y1 = Y.left,
                        y2 = Y.right,
                        color = colorviz,
                        alpha = example_df["COUNT_scaled"].iloc[counter],
                        zorder = -1,
                        lw = 0
                    )
                    # Fancy annotations cuz why not?
                    if example_df['COUNT_share'].iloc[counter] > .005:
                        text_ = ax.annotate(
                            xy = (X.right - (X.right - X.left)/2, Y.right - (Y.right - Y.left)/2),
                            text = f"{example_df['COUNT_share'].iloc[counter]:.0%}",
                            fontproperties=prop2,
                            ha = "center",
                            va = "center",
                            color = "w",
                            size = 14,
                            weight = "bold",
                            zorder = 3
                        )
                
                        text_.set_path_effects(
                            [path_effects.Stroke(linewidth=0.75, foreground="k"), path_effects.Normal()]
                        )
                    counter += 1
                ax.set_ylim(52.3,110)
                #ax.text(52.5,70, "" + PlayerSelExpData_txt.upper() + " - " + str(len(dfKK)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding colorbar
                ax9 = fig.add_axes([0.14,0.135,0.20,0.07])
                ax9.scatter(5.50,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=1.0)
                ax9.scatter(4.00,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=0.7)
                ax9.scatter(2.50,5, c=colorviz, marker='h', s=300, edgecolors='#121214', alpha=0.2)
                ax9.text(4, 0.25, '-  REMATES  +',c='w', fontproperties=prop2, fontsize=10, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
        
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

    if PlotVizSelFDData == "Passes":
        with menuoptcon02:
            OptionPlot = ['Passes Viz', 'Progressive Passes', 'Passes Into Final Third', 'Passes Into Penalty Area', 'Long Passes', 'Passes Into Half Spaces']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        with menuoptcon03:
            EfectMinSel = st.slider('Choose match time range:', 0, MaxAddMin, (0, MaxAddMin))
        with menuoptcon04:
                MetOption = ['WinStats', 'FD']
                MetOptionSel = st.selectbox('Choose color type:', MetOption)
        if MetOptionSel == 'WinStats':
            hex_list2 = ['#121214', '#D81149', '#FF0050']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#FF0050"
        if MetOptionSel == 'FD':
            hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#83E604"
        #if OptionPlot == 'Passes Map':
        #    with pltmnop13:
        #        MetOption = ['Pases Claves', 'Asistencias']
        #        MetOptionSel = st.selectbox('Seleccionar métrica:', MetOption)
        #if OptionPlot == 'Progressive Passes Map':
        #    with pltmnop13:
        #        ValorProg = st.select_slider('Valor progresivo:', 0, 10)
    
        pltmain11, pltmain12 = st.columns(2)
        with pltmain11:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            img_path = "Resources/Img/winstats_pplwhite.png"
            img = Image.open(img_path)
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            if OptionPlotSel == 'Passes Viz':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index()
                dfKKK = df
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                dfast = df[df['Event'] == 'Assists']
                dfkey = df[df['Event'] == 'Key Passes']
                dfpas = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')]
    
                #Progressive
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.75 for x in range(len(df.Beginning))]
    
                #Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)    
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                x1 = dfpas['X1']
                y1 = dfpas['Y1']
                x2 = dfpas['X2']
                y2 = dfpas['Y2']
    
                x1a = dfprog['X1']
                y1a = dfprog['Y1']
                x2a = dfprog['X2']
                y2a = dfprog['Y2']
    
                x1k = dfkey['X1']
                y1k = dfkey['Y1']
                x2k = dfkey['X2']
                y2k = dfkey['Y2']
                pitch.lines(x1, y1, x2, y2, cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True) 
                ax.scatter(x2, y2, color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)       
    
                pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)           
    
                pitch.lines(x1k, y1k, x2k, y2k, cmap=get_continuous_cmap(hex_list1), ax=ax, lw=2, comet=True, transparent=True, zorder=10) 
                ax.scatter(x2k, y2k, color="#C7B200", edgecolors='#121214', zorder=5, lw=0.5)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKK)) + " PASES COMPLETOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(52.5, 12, marker='s', color=colorviz, s=300)
                ax9.text(52.5, 2, 'PASES PROGRESIVOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(78.75, 12, marker='s', color='#C7B200', s=300)
                ax9.text(78.75, 2, 'PASES CLAVES', color='#C7B200', fontproperties=prop2, ha='center', fontsize=9)
    
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
    
            if OptionPlotSel == 'Progressive Passes':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                #dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #Progressive
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.75 for x in range(len(df.Beginning))]
    
                #Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)
                dfprog = dfprog.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='first')
                dfw = dfprog[(dfprog['Event'] == 'Successful passes') | (dfprog['Event'] == 'Key Passes') | (dfprog['Event'] == 'Assists') | (dfprog['Event'] == 'Successful open play crosses') | (dfprog['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = dfprog[(dfprog['Event'] == 'Unsuccessful passes') | (dfprog['Event'] == 'Unsuccessful open play crosses') | (dfprog['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
    
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
    
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
    
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)     
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfprog)) + " PASES PROGRESIVOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                df = dfprog.reset_index(drop=True)
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            if OptionPlotSel == 'Passes Into Final Third':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)
    
                dfwin = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dffail = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                dfKKK = df
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
    
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                pitch.lines(dfwin['X1'], dfwin['Y1'], dfwin['X2'], dfwin['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dfwin['X2'], dfwin['Y2'], color='#FF0050', edgecolors='#121214', zorder=3, lw=0.5)
                pitch.lines(dffail['X1'], dffail['Y1'], dffail['X2'], dffail['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dffail['X2'], dffail['Y2'], color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)
                ax.vlines(x=70, ymin=0, ymax=68, color='w', alpha=0.3, ls='--', zorder=-1)
                ax.add_patch(Rectangle((70, 0), 35, 68, fc="#000000", fill=True, alpha=0.7, zorder=-2))
    
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKK)) + " PASES HACIA ÚLTIMO TERCIO", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 12, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 2, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 12, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 2, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
    
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
                #pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                #ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)
            
            if OptionPlotSel == 'Passes Into Penalty Area':
                
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                # Coordenadas del cuadrilátero
                x1_cuadrilatero, y1_cuadrilatero = 88.5, 13.84
                x2_cuadrilatero, y2_cuadrilatero = 105, 13.84
                x3_cuadrilatero, y3_cuadrilatero = 88.5, 54.16
                x4_cuadrilatero, y4_cuadrilatero = 105, 54.16
    
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1 = (
                    (df['X1'] < x1_cuadrilatero) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero)      # Y1 debe ser mayor que y3_cuadrilatero
                )
    
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2 = (
                    (df['X2'] >= x1_cuadrilatero) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
    
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[condicion1 & condicion2]
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
    
                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
    
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
    
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)  
                ax.vlines(x=88.5, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=105, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(dfKKK)) + " PASES HACIA ÁREA RIVAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == "Passes Into Half Spaces":
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #df = df[(df['X1'] >= 52.5) & (df['X1'] <= 88.5)].reset_index(drop=True)
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
    
                # Coordenadas del cuadrilátero A
                x1_cuadrilatero_A, y1_cuadrilatero_A = 52.5, 43.16
                x2_cuadrilatero_A, y2_cuadrilatero_A = 88.5, 43.16
                x3_cuadrilatero_A, y3_cuadrilatero_A = 52.5, 54.16
                x4_cuadrilatero_A, y4_cuadrilatero_A = 88.5, 54.16
    
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1_A = (
                    (df['X1'] < x1_cuadrilatero_A) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero_A) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero_A) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero_A)      # Y1 debe ser mayor que y3_cuadrilatero
                )
    
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2_A = (
                    (df['X2'] >= x1_cuadrilatero_A) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero_A) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero_A) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero_A)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
    
                # Coordenadas del cuadrilátero B
                x1_cuadrilatero_B, y1_cuadrilatero_B = 52.5, 13.84
                x2_cuadrilatero_B, y2_cuadrilatero_B = 88.5, 13.84
                x3_cuadrilatero_B, y3_cuadrilatero_B = 52.5, 24.84
                x4_cuadrilatero_B, y4_cuadrilatero_B = 88.5, 24.84
    
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1_B = (
                    (df['X1'] < x1_cuadrilatero_B) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero_B) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero_B) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero_B)      # Y1 debe ser mayor que y3_cuadrilatero
                )
    
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2_B = (
                    (df['X2'] >= x1_cuadrilatero_B) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero_B) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero_B) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero_B)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
    
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[(condicion1_A & condicion2_A) | (condicion1_B & condicion2_B)].reset_index(drop=True)
    
                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
    
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)
    
                ax.vlines(x=52.5, ymin=43.16, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=88.5, ymin=43.16, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=43.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
    
                ax.vlines(x=52.5, ymin=13.84, ymax=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=88.5, ymin=13.84, ymax=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.add_patch(Rectangle((52.5, 13.84), 36, 11, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.add_patch(Rectangle((52.5, 43.16), 36, 11, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " PASES HACIA ESPACIOS INTERMEDIOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            if OptionPlotSel == 'Long Passes':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                df['Distance'] = np.sqrt((df['X2'] - df['X1'])**2 + (df['Y2'] - df['Y1'])**2)
                df = df[df['Distance'] >= 32].reset_index(drop=True)
                #df = df.drop(['Distance'], axis=1)
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " PASES LARGOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

    if PlotVizSelFDData == "Def. Actions":
        with menuoptcon02:
            OptionPlot = ['Defensive Actions Map', 'Defensive Actions Zones - Heatmap', 'Defensive Actions Bins - Heatmap', 'Territory Defensive Actions Map']
            OptionPlotSel = st.selectbox('Choose viz:', OptionPlot)
        with menuoptcon03:
            EfectMinSel = st.slider('Choose match time range:', 0, MaxAddMin, (0, MaxAddMin))
        with menuoptcon04:
                MetOption = ['WinStats', 'FD']
                MetOptionSel = st.selectbox('Choose color type:', MetOption)
        if MetOptionSel == 'WinStats':
            hex_list2 = ['#121214', '#D81149', '#FF0050']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#FF0050"
            # Definir los colores base con transparencias diferentes
            red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
            green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
            blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
            # Crear una lista de los colores y las posiciones en el colormap
            colors = [red, green, blue]
            positions = [0, 0.5, 1]
            cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)

        if MetOptionSel == 'FD':
            hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#83E604"
            # Definir los colores base con transparencias diferentes
            red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
            green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
            blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
            # Crear una lista de los colores y las posiciones en el colormap
            colors = [red, green, blue]
            positions = [0, 0.5, 1]
            # Crear el colormap continuo con transparencias
            cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
        MetricsOption = ['Defensive Actions', 'Tackles won', 'Tackles lost', 'Defensive blocks', 'Interceptions', 'Clearances', 'Recoveries']
        MetricsOptionSel = st.selectbox('Choose metrics:', MetricsOption)

        if MetricsOptionSel == 'Defensive Actions':
            df = df[(df['Event'] == 'Tackles won') | (df['Event'] == 'Tackles lost') | (df['Event'] == 'Defensive blocks') | (df['Event'] == 'Interceptions') | (df['Event'] == 'Clearances') | (df['Event'] == 'Recoveries')].reset_index(drop=True)
        else:
            df = df[df['Event'] == MetricsOptionSel].reset_index(drop=True)

        pltmain11, pltmain12 = st.columns(2)
        with pltmain11:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
    
            if OptionPlotSel == 'Defensive Actions Map':
                
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " " + MetricsOptionSel.upper() + "", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 4.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 0, 'ACCIONES\nDEFENSIVAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Defensive Actions Zones - Heatmap':
                
                path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " " + MetricsOptionSel.upper() + "", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES DEFENSIVAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Defensive Actions Bins - Heatmap':
    
                bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " " + MetricsOptionSel.upper() + "", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES DEFENSIVAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Territory Defensive Actions Map':
                
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                scaler  = StandardScaler()
                defpoints1 = df[['X1', 'Y1']].values
                defpoints2 = scaler.fit_transform(defpoints1)
                df2 = pd.DataFrame(defpoints2, columns = ['Xstd', 'Ystd'])
                df3 = pd.concat([df, df2], axis=1)
                df5=df3
                df3 = df3[df3['Xstd'] <= 1]
                df3 = df3[df3['Xstd'] >= -1]
                df3 = df3[df3['Ystd'] <= 1]
                df3 = df3[df3['Ystd'] >= -1].reset_index()
                df9 = df
                df = df3
                defpoints = df[['X1', 'Y1']].values
                #st.write(defpoints)
                hull = ConvexHull(df[['X1','Y1']])        
                ax.scatter(df9['X1'], df9['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                #Loop through each of the hull's simplices
                for simplex in hull.simplices:
                    #Draw a black line between each
                    ax.plot(defpoints[simplex, 0], defpoints[simplex, 1], '#BABABA', lw=2, zorder = 1, ls='--')
                ax.fill(defpoints[hull.vertices,0], defpoints[hull.vertices,1], colorviz, alpha=0.7)
                meanposx = df9['X1'].mean()
                meanposy = df9['Y1'].mean()
                ax.scatter(meanposx, meanposy, s=1000, color="w", edgecolors=colorviz, lw=2.5, zorder=25, alpha=0.95)
                names = PlayerSelExpData.split()
                iniciales = ""
                for name in names:
                   iniciales += name[0] 
                #names_iniciales = names_iniciales.squeeze().tolist()
                ax.text(meanposx, meanposy, iniciales, color='k', fontproperties=prop2, fontsize=13, zorder=34, ha='center', va='center')
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " " + MetricsOptionSel.upper() + "", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -0.5, 'ACCIONES \nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

    if PlotVizSelFDData == "Possession":
        with menuoptcon02:
            OptionPlot = ['Dribbles Map', 'Dribbles Zones - Heatmap', 'Offensive Actions Map']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        with menuoptcon03:
            EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
        with menuoptcon04:
            MetOption = ['WinStats', 'FD']
            MetOptionSel = st.selectbox('Choose color type:', MetOption)
        
        if MetOptionSel == 'WinStats':
            hex_list2 = ['#121214', '#D81149', '#FF0050']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#FF0050"
            # Definir los colores base con transparencias diferentes
            red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
            green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
            blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
            # Crear una lista de los colores y las posiciones en el colormap
            colors = [red, green, blue]
            positions = [0, 0.5, 1]
            # Crear el colormap continuo con transparencias
            cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
        if MetOptionSel == 'FD':
            hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#83E604"
            # Definir los colores base con transparencias diferentes
            red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
            green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
            blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
            # Crear una lista de los colores y las posiciones en el colormap
            colors = [red, green, blue]
            positions = [0, 0.5, 1]
            # Crear el colormap continuo con transparencias
            cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
        
        pltmain11, pltmain12 = st.columns(2)
        with pltmain11:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            df = df[(df['Event'] == 'Successful dribbles') | (df['Event'] == 'Unsuccessful dribbles') | (df['Event'] == 'Fouls won') | (df['Event'] == 'Fouls lost')].reset_index(drop=True)
    
            
            if OptionPlotSel == 'Dribbles Map':
                
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " DRIBBLES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 4.5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 0, 'DRIBBLES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
    
            if OptionPlotSel == 'Dribbles Zones - Heatmap':
                
                path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                ax.text(52.5,70, "" + PlayersFDSel.upper() + " - " + str(len(df)) + " " + MetricsOptionSel.upper() + "", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  DRIBBLES  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                dfKK = df
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        
        
