import xgboost as xgb
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


url_data = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.data"
url_names = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.names"

urlretrieve(url_data, "agaricus-lepiota.data")
urlretrieve(url_data, "agaricus-lepiota.names")


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

agar_train, agar_test = train_test_split(agar, test_size=.8, random_state=1999)


agar_train_mat = xgb.DMatrix(agar_train.drop("target", 1), label=agar_train["target"])
agar_test_mat = xgb.DMatrix(agar_test.drop("target", 1), label=agar_test["target"])

parametros = {"booster":"gbtree", "max_depth": 3, "eta": 0.3, "objective": "binary:logistic"}
rondas = 10
evaluacion = [(agar_test_mat, "eval"), (agar_train_mat, "train")]

modelo = xgb.train(parametros, agar_test_mat, rondas, evaluacion)


prediccion = modelo.predict(agar_test_mat)
prediccion = [1 if i > .85 else 0 for i in prediccion]

def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = metricas(agar_test["target"], prediccion)
[print(i) for i in metricas]


modelo.save_model("modelo.model")

modelo_cargado = xgb.Booster()
modelo_cargado.load_model("modelo.model")

modelo_cargado.predict(agar_test_mat)