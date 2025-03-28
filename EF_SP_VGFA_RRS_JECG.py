import re
import nltk 
import pandas as pd
import numpy as np
import plotly.express as px

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

# categorias_edificios = {
#     "departamento": ["departa", "edifici departa", "edifici aparta", "unid departa", "edifici pi",
#                      "departa ranch san lorenz"],
#     "residencial": ["residenci"],
#     "vecindad": ["vecind", "edif antigu tip vecind"],
#     "condominio": ["condomini"],
#     "multifamiliar": ["multifamili", "unid habitacion"],
#     "oficina": ["oficin", "pi oficin comerci mixt"],
#     "comercio": ["comerci", "hotel", "hotel royal reform"],
#     "escuela": ["escuel", "escuel primari", "escuel primari koweit", 
#                 "escuel secundari", "escuel secundari cuauhtemoc", 
#                 "escuel criminolog", "facult medicin unam", "tec monterrey cdmx",  
#                 "escol"],
#     "hospital": ["hospit", "hospitalclin", "hospit ciud mexic belisari dominguez", 
#                  "hospit gener dr manuel gea gonzalez"],
#     "clínica": ["clinic pren", "laboratori", "almacen medica"],
#     "gobierno": ["predi gobiern abandon", "edifici imss"],
#     "fábrica": ["fabric", "tall"],
#     "barda": ["bard", "post luz"],
#     "infraestructura": ["puent", "banquet", "call", "call callejon", "via public", "autop"],
#     "campamento": ["campament"],
#     "asilo": ["asil priv"],
#     "aeropuerto": ["aeropuert internacion ciud mexic"],
#     "estación": ["estacion"],
# }

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

# def normalizar_edificio(text, edificios):
#     for edificio, palabras_clave in edificios.items():
#         if any(palabra in text for palabra in palabras_clave):
#             return edificio
#     return 'Otro'

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

# data['edificio_procesado'] = data['lugar'].apply(preprocess_text)
# data['edificio_procesado'] = data['edificio_procesado'].apply(remove_stopwords)
# data['edificio_procesado'] = data['edificio_procesado'].apply(stem_words)
# data['edificio_normalizado'] = data['edificio_procesado'].apply(lambda x: normalizar_edificio(x, categorias_edificios))

relevant_data = data[['lat', 'lon', 'tipo_daño_clasificado', 'riesgo_categorizado', 'delegacion_normalizada', 'escala_daño']].dropna()


x = relevant_data[['lat', 'lon']]
y = relevant_data['riesgo_categorizado']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Primeras filas de X_train normalizado:")
print(X_train_scaled[:5])

print(f"Tamaño de X_train: {X_train.shape}, X_test: {X_test.shape}")

print(f"Mínimos de cada variable: {X_train_scaled.min(axis=0)}, Máximos: {X_train_scaled.max(axis=0)}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
accuracy = knn.score(X_test_scaled, y_test)


print(f"Precisión del modelo KNN: {accuracy:.2f}")


# nuevas_coordenadas = [[19.432608, -99.133209], [19.300000, -99.200000]] 
# nuevas_coordenadas_scaled = scaler.transform(nuevas_coordenadas)

# predicciones = knn.predict(nuevas_coordenadas_scaled)
# print(f"Predicción para las nuevas coordenadas {nuevas_coordenadas}: {predicciones}")

k = relevant_data['delegacion_normalizada'].nunique()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
relevant_data['cluster'] = kmeans.fit_predict(relevant_data[['lat', 'lon']])


print("Centroides de los clusters:")
print(kmeans.cluster_centers_)

print("\nDistribución de los clusters:")
print(relevant_data['cluster'].value_counts())

fig = px.scatter_mapbox(
    relevant_data,
    lat='lat',
    lon='lon',
    color='cluster',
    hover_name='riesgo_categorizado',
    hover_data=['escala_daño', 'delegacion_normalizada'],
    color_continuous_scale="plasma",
    zoom=10,
    title="Clusters de Daño por K-Means"
)

# Configuración del mapa
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=10,
    mapbox_center={"lat": relevant_data['lat'].mean(), "lon": relevant_data['lon'].mean()}
)

# Mostrar el mapa
fig.show()

# scaler = MinMaxScaler()
# relevant_data[['lat', 'lon']] = scaler.fit_transform(relevant_data[['lat', 'lon']])

# k = 4
# kmeans = KMeans(n_clusters=k, random_state=42, n_init = 10)
# relevant_data['cluster'] = kmeans.fit_predict(relevant_data[['lat', 'lon']])

# mapeo_riesgo = {
#     0: 'Sin riesgo',
#     1: 'Bajo',
#     2: 'Moderado',
#     3: 'Alto'
# }

# relevant_data['Riesgo_kmeans'] = relevant_data['cluster'].map(mapeo_riesgo)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(relevant_data[ ['lat', 'lon']], relevant_data['escala_daño'])

# def predecir_riesgo(lat, lon):
#     coordenadas = scaler.transform([[lat, lon]])
#     dist, _ = knn.kneighbors(coordenadas)

#     if dist.mean() > 0.2:
#         return 'Sin riesgo'
#     return knn.predict(coordenadas)[0]


# fig = px.scatter_mapbox(relevant_data,
#                         lat = 'lat',
#                         lon = 'lon',
#                         color = 'Riesgo_kmeans',
#                         hover_name = 'tipo_daño_clasificado',
#                         mapbox_style = 'carto-positron',
#                         zoom = 10,
#                         title = 'Mapa de riesgo de daños en la CDMX',
# )

# fig.show()