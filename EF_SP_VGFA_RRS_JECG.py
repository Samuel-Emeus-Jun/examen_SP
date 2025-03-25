import re
import nltk 
import pandas as pd


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

data = pd.read_excel('SISTEMAS PREDICTIVOS.xlsx')

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

categorias_edificios = {
    "departamento": ["departa", "edifici departa", "edifici aparta", "unid departa", "edifici pi",
                     "departa ranch san lorenz"],
    "residencial": ["residenci"],
    "vecindad": ["vecind", "edif antigu tip vecind"],
    "condominio": ["condomini"],
    "multifamiliar": ["multifamili", "unid habitacion"],
    "oficina": ["oficin", "pi oficin comerci mixt"],
    "comercio": ["comerci", "hotel", "hotel royal reform"],
    "escuela": ["escuel", "escuel primari", "escuel primari koweit", 
                "escuel secundari", "escuel secundari cuauhtemoc", 
                "escuel criminolog", "facult medicin unam", "tec monterrey cdmx",  
                "escol"],
    "hospital": ["hospit", "hospitalclin", "hospit ciud mexic belisari dominguez", 
                 "hospit gener dr manuel gea gonzalez"],
    "clínica": ["clinic pren", "laboratori", "almacen medica"],
    "gobierno": ["predi gobiern abandon", "edifici imss"],
    "fábrica": ["fabric", "tall"],
    "barda": ["bard", "post luz"],
    "infraestructura": ["puent", "banquet", "call", "call callejon", "via public", "autop"],
    "campamento": ["campament"],
    "asilo": ["asil priv"],
    "aeropuerto": ["aeropuert internacion ciud mexic"],
    "estación": ["estacion"],
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

def normalizar_edificio(text, edificios):
    for edificio, palabras_clave in edificios.items():
        if any(palabra in text for palabra in palabras_clave):
            return edificio
    return 'Otro'

data['tipo_daño_procesado'] = data['tipo_daño'].apply(preprocess_text)
data['tipo_daño_procesado'] = data['tipo_daño_procesado'].apply(remove_stopwords)
data['tipo_daño_procesado'] = data['tipo_daño_procesado'].apply(stem_words)
data['tipo_daño_clasificado'] = data['tipo_daño_procesado'].apply(lambda x: clasificar_daño(x, categorias_daños))

data['delegacion_procesada'] = data['delegacion'].apply(preprocess_text)
data['delegacion_procesada'] = data['delegacion_procesada'].apply(remove_stopwords)
data['delegacion_procesada'] = data['delegacion_procesada'].apply(stem_words)
data['delegacion_normalizada'] = data['delegacion_procesada'].apply(lambda x: normalizar_delegacion(x, categorias_municipios))

data['edificio_procesado'] = data['lugar'].apply(preprocess_text)
data['edificio_procesado'] = data['edificio_procesado'].apply(remove_stopwords)
data['edificio_procesado'] = data['edificio_procesado'].apply(stem_words)
data['edificio_normalizado'] = data['edificio_procesado'].apply(lambda x: normalizar_edificio(x, categorias_edificios))
# print(data['tipo_daño_procesado'])