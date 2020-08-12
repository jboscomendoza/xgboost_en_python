import xgboost as xgb
import pandas as pd
from numpy import array
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


url_data = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.data"
url_names = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.names"

urlretrieve(url_data, "agaricus-lepiota.data")
urlretrieve(url_names, "agaricus-lepiota.names")


def ver_contenido(ruta, mode="r", lineas=10):
    with open(ruta) as archivo:
        contenido = archivo.readlines()
        [print(i) for i in contenido[:lineas]]

ver_contenido("agaricus-lepiota.data")
ver_contenido("agaricus-lepiota.names")

nombres = [
    "target", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", 
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", 
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", 
    "veil_color", "ring_number", "ring_type", "spore_print_color", "population",
    "habitat"
  ]

agar = pd.read_csv("agaricus-lepiota.data", names=nombres)
agar.head(5)


def str_a_num(df):
    for col in df:
        original = df[col].unique()
        reemplazo = list(range(len(original)))
        mapa = dict(zip(original, reemplazo))
        df[col] = df[col].replace(mapa)
    return(df)

str_a_num(agar)


agar_train, agar_test = train_test_split(agar, test_size=.3, random_state=1996)

agar_train_mat = xgb.DMatrix(agar_train.drop("target", 1), label=agar_train["target"])
agar_test_mat = xgb.DMatrix(agar_test.drop("target", 1), label=agar_test["target"])

parametros = {"booster":"gbtree", "max_depth": 2, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
rondas = 10
evaluacion = [(agar_test_mat, "eval"), (agar_train_mat, "train")]

modelo = xgb.train(parametros, agar_test_mat, rondas, evaluacion)


prediccion = modelo.predict(agar_test_mat)
prediccion = [1 if i > .6 else 0 for i in prediccion]

def metricas(objetivo, predict):
    matriz_conf = confusion_matrix(objetivo, predict)
    score = accuracy_score(objetivo, predict)
    reporte = classification_report(objetivo, predict)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = metricas(agar_test["target"], prediccion)
[print(i) for i in metricas]


parametros_02 = {"booster":"gbtree", "max_depth": 4, "eta": .3, "objective": "binary:logistic", "nthread":2}
rondas_02 = 100
evaluacion = [(agar_test_mat, "eval"), (agar_train_mat, "train")]

modelo_02 = xgb.train(parametros_02, agar_test_mat, rondas_02, evaluacion, early_stopping_rounds=10)


prediccion_02 = modelo_02.predict(agar_test_mat)
prediccion_02 = [1 if i > .5 else 0 for i in prediccion_02]

metricas_02 = metricas(agar_test["target"], prediccion_02)
[print(i) for i in metricas_02]


modelo.save_model("modelo.model")

modelo_importado = xgb.Booster()
modelo_importado.load_model("modelo.model")
modelo_importado.predict(agar_test_mat)

from numpy import array

nuevo = array([
    [2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
nuevo_mat = xgb.DMatrix(nuevo, feature_names = nombres[1:])
modelo_02.predict(nuevo_mat)