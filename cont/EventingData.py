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
st.header("EVENTING DATA")

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

st.divider()


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
    st.write("This is inside the container")
    menuoptcon01, menuoptcon02, menuoptcon03, menuoptcon04 = st.columns(4)
    with menuoptcon01:
        PlotVizSelFDData = st.selectbox("Choose Metric", ['Actions', 'Passes', 'Shots', 'Def. Actions', 'Possession'])
   
#container = st.container(border=True)
#container.write("This is inside the container")
#st.write("This is outside the container")
# Now insert some more in the container
#container.write("This is inside too")
    df['EfectiveMinute'] = df['EfectiveMinute'].astype(int)
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
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
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
