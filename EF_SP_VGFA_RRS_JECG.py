import re
import nltk 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

data = pd.read_excel('SISTEMAS PREDICTIVOS.xlsx')

data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
data['lon'] = pd.to_numeric(data['lon'], errors='coerce')

data = data.dropna(subset=['lat', 'lon'])

stemmer = SnowballStemmer('spanish')
stemmer_p = PorterStemmer()
stop_words = set(stopwords.words('spanish'))


categorias_daños = {
    "daños menores": ["fisur", "fisur dañ","dañ menor", "pi levant", "post inclin", "fug ga", "tub rot", "arbol caid", "griet superfici"],
    "grietas": ["griet dañ estructur edifici previ deterior", "griet", "griet mayor", "griet estructur", "griet mur interior", "griet column mur desplaz mur", "griet pi rode edif"],
    "daños mayores": ["dañ", "dañ especif", "dañ mayor", "apunt caer"],
    "derrumbe": ["derrumb", "derrumb parcial", "derrumb total", "derrumb bard", "derrumb mur divisori", "derrumb barb", "derrumn", "desplom ruptur cimient"],
    "hundimientos": ["hundimi", "hundimi pi", "hundimi inclin", "hundimmi", "edifici inclin"]
}

categorias_municipios = {
    "Azcapotzalco": ["azcapotzalc", "atzcapotzalc", "axcapotzalc"],
    "Coyoacán": ["coyoacan", "coyoac"],
    "Cuajimalpa de Morelos": ["cuajimalp", "cuajimalp morel", "cuajimalp col contad"],
    "Gustavo A. Madero": ["gustav mader", "gust mader", "gam", "gustavoy mader"],
    "Iztacalco": ["iztacalc", "rey iztac"],
    "Iztapalapa": ["iztapalap", "iztalap", "iztap", "iztapalap san lorenz tezonc", "ixtapalap"],
    "La Magdalena Contreras": ["magdalen contrer", "magdalen salin"],
    "Milpa Alta": ["milp alta"],
    "Álvaro Obregón": ["alvar obregon", "alvar obtegon", "obregon", "sant fe alvar obregon",],
    "Tláhuac": ["tlahuac", "tlauac", "tkahuac", "tlhuac cd mx", ],
    "Tlalpan": ["tlalp"],
    "Xochimilco": ["xochimilc"],
    "Benito Juárez": ["benit juarez", "benit jurez", "benit justez", "benit juar", "benit juarez col jo insurg", "bj"],
    "Cuauhtémoc": [
        "cuauhtemoc", "cuahutemoc", "cuathemoc", "cuauht", "cuahuhtemoc",
        "cuhuactemoc", "cuauhtemoc cdad mexic", "cuauhtemocc", "cuautemoch",
        "centr histor cuauhtemoc", "coahutemoc"
    ],
    "Miguel Hidalgo": [
        "miguel hidalg", "miguel hidag", "mig hidalg", 
        "miguel hidalg san miguel chapultepec ii seccion", "m hidalg"
    ],
    "Venustiano Carranza": [
        "venustian carranz", "venustian", "venu tian carranz", "v carranz"
    ],
    "Ecatepec de Morelos": [
        "ecatepec", "ecatepec morel", "ecatepec mexic", "municipi ecatepec",
        "nuev lare ecatepec edo mex ecatepec morel"
    ],
    "Nezahualcóyotl": [
        "nezahualcoyotl", "nezahualcoyolt", "nezahualcoyotl mexic",
        "municipi nezahualcoyotl"
    ],
    "Naucalpan de Juárez": ["naucalp", "naucalp juarez", "naucalp edod mex"],
    "Atizapán de Zaragoza": ["atizap zarag", "atizap zaragoz"],
    "Tlalnepantla de Baz": ["tlalnepantl", "tlalnepantl baz", "municipi tlalnepantl", "tlanepantl",],
    "Coacalco de Berriozábal": ["coacalc", "coacalc edomex"],
    "Chimalhuacán": ["chimalhuacan", "chimalhuac", "chimalhuac edo mex"],
    "Cuautitlán": ["cuautitlan"],
    "Cuautitlán Izcalli": ["cuautitlan izcal", "cuautitlan izcal edo mex"],
    "Huixquilucan": ["huixquiluc", "huixquiluc mexic", "huixquiluc edomex"],
    "Tecámac": ["tecamac", "tecamac felip villanuev"],
    "Tultitlán": ["tultitl", "tultitl marian escob mexic", "municipi tultitl"],
    "Chicoloapan": ["chicoloap"],
    "Texcoco": ["texcoco"],
    "La Paz": ["paz"],
    "Melchor Ocampo": ["melchor ocamp"],
    "Nicolás Romero": ["nicola romer", "municipi nicol romer edo mex"],
    "Teotihuacán": ["teotihuacan"],
    "Tezoyuca": ["tezoyuca"],
    "Zumpango": ["zumpango"],
    "Valle de Chalco Solidaridad": ["vall chalc", "chalc", ]
}

mapa_tipo_daño = {
    "daños menores": 0,
    "grietas": 1,
    "daños mayores": 2,
    "derrumbe": 3,
    "hundimientos": 4
}


def preprocess_text(text):
    text = str(text)  
    text = text.lower() 
    text = re.sub(r'[^a-záéíóúñ ]', '', text)  
    tokens = word_tokenize(text)  
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Stemming y remover stopwords
    return ' '.join(tokens)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def stem_words(text):
    words = text.split()
    stemmed_words = [stemmer_p.stem(word) for word in words]
    return ' '.join(stemmed_words)

def clasificar_daño(text, categorias):
    for categoria, palabras_clave in categorias.items():
        if any(palabra in text for palabra in palabras_clave):
            return categoria
    return 'No Clasificado'

def normalizar_delegacion(text, delegaciones):
    for delegacion, palabras_clave in delegaciones.items():
        if any(palabra in text for palabra in palabras_clave):
            return delegacion
    return 'Provincia'

def categorizar_riesgo(valor):
    if valor == 0:
        return 'bajo'
    elif valor == 1:
        return 'medio'
    else:
        return 'alto'

def pedir_coordenadas():
    try:
        input_str = input("Introduce las coordenadas (latitud, longitud) separadas por comas: ")
        lat_str, lon_str = input_str.split(",")
        print([float(lat_str), float(lon_str)])
        return [[float(lat_str), float(lon_str)]] 
    except ValueError:
        print("Error: Debes introducir exactamente dos valores separados por comas.")
        return None #pedir_coordenadas()
    


##LIMPIEZA BBDD

data['tipo_daño_procesado'] = data['tipo_daño'].apply(preprocess_text)
data['tipo_daño_procesado'] = data['tipo_daño_procesado'].apply(remove_stopwords)
data['tipo_daño_procesado'] = data['tipo_daño_procesado'].apply(stem_words)
data['tipo_daño_clasificado'] = data['tipo_daño_procesado'].apply(lambda x: clasificar_daño(x, categorias_daños))

data['escala_daño'] = data['tipo_daño_clasificado'].map(mapa_tipo_daño)
data['riesgo_categorizado'] = data['escala_daño'].apply(categorizar_riesgo)


data['delegacion_procesada'] = data['delegacion'].apply(preprocess_text)
data['delegacion_procesada'] = data['delegacion_procesada'].apply(remove_stopwords)
data['delegacion_procesada'] = data['delegacion_procesada'].apply(stem_words)
data['delegacion_normalizada'] = data['delegacion_procesada'].apply(lambda x: normalizar_delegacion(x, categorias_municipios))

data = data.query('delegacion_normalizada != "Provincia"')


##KNN TRAINING

relevant_data = data[['lat', 'lon', 'tipo_daño_clasificado', 'riesgo_categorizado', 'delegacion_normalizada', 'escala_daño']].dropna()

x = relevant_data[['lat', 'lon']]
y = relevant_data['escala_daño']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("Primeras filas de X_train normalizado:")
# print(X_train_scaled[:5])
# print(f"Tamaño de X_train: {X_train.shape}, X_test: {X_test.shape}")
# print(f"Mínimos de cada variable: {X_train_scaled.min(axis=0)}, Máximos: {X_train_scaled.max(axis=0)}")

relevant_data[['lat_normalizada', 'lon_normalizada']] = scaler.fit_transform(relevant_data[['lat', 'lon']])
#print(relevant_data[['lat_normalizada', 'lon_normalizada']].head())

knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
knn.fit(X_train_scaled, y_train)
accuracy = knn.score(X_test_scaled, y_test)

##PRUEBAS MODELO KNN

print(f"Precisión del modelo KNN: {accuracy:.2f}")



nuevas_coordenadas = [[19.3367846, -99.2284202]] #pedir_coordenadas()
nuevas_coordenadas_scaled = scaler.transform(nuevas_coordenadas)

predicciones = knn.predict(nuevas_coordenadas_scaled)
print(f"Predicción para las nuevas coordenadas {nuevas_coordenadas}: {predicciones}")


from sklearn.metrics import classification_report

y_pred = knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred))


##DISEÑO DE HEATMAP


fig = px.scatter_mapbox(
    relevant_data,
    lat='lat',
    lon='lon',
    color='escala_daño',  
    size= [10] * len(relevant_data),
    hover_name='tipo_daño_clasificado',
    hover_data={'riesgo_categorizado': True, 'escala_daño': True},
    center={'lat': 19.4, 'lon': -99.2},
    zoom=10,
    mapbox_style='carto-positron',
    color_continuous_scale='Turbo',
    opacity = 0.7,
)


fig.add_trace(go.Scattermapbox(
    lat=relevant_data['lat'],
    lon=relevant_data['lon'],
    mode='markers',
    marker=dict(size=8, color='rgba(0,0,0,0)'),
    text = relevant_data['tipo_daño_clasificado'],
    hoverinfo='text'
))


fig.update_traces(
    marker=dict(sizemode='area', sizemin=3, sizeref=2.0),
    hovertemplate="<b>%{hovertext}</b><extra></extra>"
)


#fig.write_html("mapa.html")
fig.show()


# fig = px.density_map(
#     relevant_data,
#     lat = 'lat',
#     lon = 'lon',
#     z = 'escala_daño',
#     radius = 10,
#     center = {'lat': 19.4, 'lon' : -99.2},
#     zoom = 10,
#     map_style = 'carto-darkmatter',
#     color_continuous_scale = 'Turbo',
        
# )

# fig.data[0].hoverinfo = 'skip'


# fig.update_coloraxes(
#     colorbar_title = 'Escala de Daño',
#     colorbar_tickvals = [0, 2, 4],
#     colorbar_ticktext = ['Bajo', 'Medio', 'Alto'],
# )


# fig.add_scattermap(
#     lat=[nuevas_coordenadas[0][0]],
#     lon=[nuevas_coordenadas[0][1]], 
#     mode='markers',
#     marker=dict(
#         size= 25, 
#         color='white',  
#         symbol='circle',    
#         ),
#     hovertext = 'Coordenadas del usuario',
#     #showlegend = False,
# )

# fig.add_scattermap(
#     lat=relevant_data['lat'], 
#     lon=relevant_data['lon'],
#     mode='markers',
#     marker=dict(size=15, color='Red'),
#     hoverinfo='text',
#     text = relevant_data[['tipo_daño_clasificado', 'riesgo_categorizado']].apply(lambda x: f"Daño recibido: {x[0]}<br>Riesgo de derrumbe: {x[1]}", axis=1),
#     #showlegend=False,
# )

# fig.update_layout(
#     title = 'Mapa de Calor de Daños en la CDMX',
#     map_layers=[{"below": "traces"}])


# fig.show()







##PRUEBA DE INERCIA PARA ENCONTRAR EL CODO (WEY, ESTO SOLO TIENE SENTIDO SI ERES IED)

# inertia = []
# k_range = range(2, 30)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(relevant_data[['lat_normalizada', 'lon_normalizada']])
#     inertia.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(k_range, inertia, marker='o')
# plt.title('Método del codo para determinar el número óptimo de clusters')
# plt.xlabel('Número de clusters (k)')
# plt.ylabel('Inercia')
# plt.show()
