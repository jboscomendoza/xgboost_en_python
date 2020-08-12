
# Tutorial: XGBoost en Python

XGBoost (Extreme Gradient Boosting), es uno de los algoritmos de machine learning de tipo supervisado más usados en la actualidad.

Este algoritmo se caracteriza por obtener buenos resultados de predicción con relativamente poco esfuerzo, en muchos casos equiparables o mejores que los devueltos por modelos más complejos computacionalmente, en particular para problemas con datos heterogéneos.

XGBoost es una herramienta muy útil para un data scientist que cuenta con implementaciones para diferentes lenguajes y entornos de programación.

En este artículo revisaremos la implementación de XGBoost en Python 3. Veremos cómo preparar los datos para usar este algoritmo, sus hiper parámetros básicos, una manera sencilla de evaluar sus resultados y cómo exportar un modelo predictivo una vez que lo hemos entrenado.

Pero antes, una breve introducción a XGBoost.

# Una introducción informal a XGBoost

XGBoost Extreme Gradient Boosting es un algoritmo predictivo supervisado que utiliza el principio de *boosting*.

La idea detrás del boosting es generar múltiples modelos de predicción “débiles” secuencialmente, y que cada uno de estos tome los resultados del modelo anterior, para generar un modelo más “fuerte”, con mejor poder predictivo y mayor estabilidad en sus resultados.

Para conseguir un modelo más fuerte a partir de estos modelos débiles, se emplea un algoritmo de optimización, este caso Gradient Descent (descenso de gradiente).

Durante el entrenamiento, los parámetros de cada modelo débil son ajustados iterativamente tratando de encontrar el mínimo de una función objetivo, que puede ser la proporción de error en la clasificación, el área bajo la curva (AUC), la raíz del error cuadrático medio (RMSE) o alguna otra.

Cada modelo es comparado con el anterior. Si un nuevo modelo tiene mejores resultados, entonces se toma este como base para realizar modificaciones. Si, por el contrario, tiene peores resultados, se regresa al mejor modelo anterior y se modifica ese de una manera diferente. Qué tan grandes son los ajustes de un modelo a otro es uno de los hiper parámetros que debe definir el usuario.

Este proceso se repite hasta llegar a un punto en el que la diferencia entre modelos consecutivos es insignificante, lo cual nos indica que hemos encontrado el mejor modelo posible, o cuando se llega al número de iteraciones máximas definido por el usuario.

XGBoost usa como sus modelos débiles árboles de decisión de diferentes tipos, que pueden ser usados para tareas de clasificación y de regresión, por lo que no está de más dar un repaso a los fundamentos de este tipo de modelos para sacar el mayor provecho del algoritmo.

Si quieres conocer más sobre este algoritmo, puedes leer definiciones más formales que incluyen discusión sobre su implementación en los siguientes artículos:

* [Greedy Function Approximation: A Gradient Boosting Machine (Friedman, 2001)](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451)
* [Stochastic Gradient Boosting (Friedman, 1999)](https://astro.temple.edu/~msobel/courses_files/StochasticBoosting(gradient).pdf)

Ahora veamos cómo usar XGBoost en Python.

# Instalación

Lo primero que necesitamos es instalar los paquetes que usaremos con `pip`.

* **xgboost**: la implementación de este algoritmo para Python.
* **pandas**: la librería por excelencia para analizar y manipular datos de manera eficiente en Python.
* **sci kit learn (sklearn)**: sólo usaremos algunas funciones de esta librería que contiene herramientas esenciales para tareas de machine learning.
* **urllib**: parte de las librerías standard de Python, usaremos una función para descargar archivos.
* **numpy**: permite computo científico en Python, otra librería de la que sólo usaremos una función.

``` shell
pip install xgboost pandas sklearn
```

Cargamos las librerías y componentes que usaremos con `import`.

``` python
import pandas as pd
import xgboost as xgb
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from urllib.request import urlretrieve
```

Las versiones de las librerias usadas en este artículo son las siguientes:

* pandas 1.0.2
* xgboost 1.1.1
* numpy 1.18.2
* scikit-learn 0.22
* urllib3 1.24.3

# Datos que usaremos

El conjunto de datos que usaremos es conocido como *Agraricus*. Contiene características de diferentes hongos y el objetivo de la tarea de clasificación es predecir a partir de ellas si son venenosos o no.

Como en la práctica es común que tengas que lidiar con datos en formatos no convencionales que requieren preparación antes de ser usables por un algoritmo de machine learning, en lugar de usar la versión de estos datos incluida en el paquete xgboost y que ya está lista para usar, trabajaremos con una versión de estos mismos datos que requiere preparación. 

Generalmente, la preparación de datos es la etapa del flujo de trabajo de machine learning que requiere más tiempo y esfuerzo. Si alguna vez te encuentras con un csv o una tabla de una base de datos que ya contenga todo lo que necesitas, perfectamente estructurado, básicamente has sacado la lotería.

La versión de los datos Agaricus que usaremos está disponible en el Machine Learning Repository de UCI.

* https://archive.ics.uci.edu/ml/datasets/Mushroom

He copiado los datos a un repositorio de github para asegurar que estés usando la misma versión que aparece en este artículo. Son dos archivos en total, uno con extensión .data que contiene los datos de los hongos, y otro de extensión .names que incluye una descripción de ellos.

Descargamos ambos archivos a nuestra carpeta de trabajo usando la función `urlretrive()` de `urllib`

```python
url_data = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.data"
url_names = "https://raw.githubusercontent.com/jboscomendoza/xgboost_en_python/master/agaricus-lepiota.names"

urlretrieve(url_data, "agaricus-lepiota.data")
urlretrieve(url_names, "agaricus-lepiota.names")
```

Ya que contamos con los datos que usaremos, procedemos a explorarlos.

# Exploración de los datos

Podemos dar una mirada al contenido de nuestros archivos con algún procesador de texto externo, como notepad++ o gedit, pero también podemos hacer esto directamente en Python con el método `readlines()`.

Para esto, crearemos una pequeña función que abrirá los archivos y mostrará sus primeras diez líneas en la terminal con `print()`. 

Le daremos como argumentos la ruta del archivo, el modo en que será abierto y cuántas líneas queremos ver, con diez por defecto.

``` python
def ver_contenido(ruta, mode="r", lineas=10):
    with open(ruta) as archivo:
        contenido = archivo.readlines()
        [print(i) for i in contenido[:lineas]]
```

Veamos los primeros renglones del archivo *agaricus-lepiota.data*.

``` python
ver_contenido("agaricus-lepiota.data")
```
```
p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u

e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g

e,b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m

p,x,y,w,t,p,f,c,n,n,e,e,s,s,w,w,p,w,o,p,k,s,u

e,x,s,g,f,n,f,w,b,k,t,e,s,s,w,w,p,w,o,e,n,a,g

e,x,y,y,t,a,f,c,b,n,e,c,s,s,w,w,p,w,o,p,k,n,g

e,b,s,w,t,a,f,c,b,g,e,c,s,s,w,w,p,w,o,p,k,n,m

e,b,y,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,s,m

p,x,y,w,t,p,f,c,n,p,e,e,s,s,w,w,p,w,o,p,k,v,g

e,b,s,y,t,a,f,c,b,g,e,c,s,s,w,w,p,w,o,p,k,s,m
```

Podemos ver que los datos se encuentran en una estructura tabular, con columnas separada por comas. Para fines prácticos, es equivalente a un archivo csv pero con una extensión diferente. Eso son buenas noticias, pues podremos usar las mismas funciones que para un archivo csv sin mayor problema.

Sin embargo, no tenemos los nombres de las columnas.

Aunque este paso no es estrictamente necesario, vale la pena obtener el nombre de cada columna, es decir, de las variables o features de nuestros datos.

Es frecuente que en la práctica tengas que trabajar con datos a los que se les ha ocultado intencionalmente el nombre de los features por seguridad o confidencialidad, sin embargo, contar con los nombres de las variables es útil entender nuestros datos y así realizar un buen análisis de ellos.

Además, en nuestro caso es esencial conocer cuál de las columnas es la variable objetivo, la que nos dice si un hongo como venenoso o no. Sin ella no podríamos continuar con nuestra tarea de clasificación.

Para nuestra buena suerte, estos se encuentran en el archivo *agaricus-lepiota.names*. 

Veamos su contenido.

```python
ver_contenido("agaricus-lepiota.names")
```

```
1. Title: Mushroom Database



2. Sources:

    (a) Mushroom records drawn from The Audubon Society Field Guide to North

        American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred

        A. Knopf

    (b) Donor: Jeff Schlimmer (Jeffrey.Schlimmer@a.gp.cs.cmu.edu)        

    (c) Date: 27 April 1987



3. Past Usage:
```

De todas maneras, nos vemos obligados a abrir este archivo con un procesador de textos y extraer de allí los nombres de cada columna.

También podemos invertir un largo rato definiendo una función ad hoc para leer específicamente este archivo y obtener los nombres de columna programáticamente. Pero, hay que ser prácticos y en este caso unos cuantos minutos haciendo un poco de copy-paste es la solución más eficiente para lo que necesitamos.

Al realizar lo anterior, encontramos que la variable objetivo es la primera columna con el nombre *target*.

Los nombres de todas las columnas son los siguientes, que guardaremos en una lista.

``` python
nombres = [
    "target", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", 
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", 
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", 
    "veil_color", "ring_number", "ring_type", "spore_print_color", "population",
    "habitat"
  ]
```

Ahora que tenemos la certeza que nuestros datos tienen forma tabular, los podemos leer con la función `read_csv()` de `pandas` con el argumento `names=nombres`, lo cual nos dará como resultado un *data frame* al que llamaremos `agar`.

``` python
agar = pd.read_csv("agaricus-lepiota.data", names=nombres)
```

Veamos el resultado con el método `head()`:

```python
agar.head(5)
```
``` 
  target cap_shape cap_surface  ... spore_print_color population habitat
0      p         x           s  ...                 k          s       u
1      e         x           s  ...                 n          n       g
2      e         b           s  ...                 n          n       m
3      p         x           y  ...                 k          s       u
4      e         x           s  ...                 n          a       g
```

Luce correcto, aunque tenemos cadenas de texto en cada celda en lugar de valores numéricos. De acuerdo al contenido del archivo agaricus-lepiota.names, cada columna representa un rasgo de los hongos que deseamos clasificar, así que cada letra es una categoría discreta.

Esto lo que cambiar para poder entrenar un modelo, porque XGBoost requiere matrices numéricas para funcionar, así que continuamos con la preparación de los datos.


# Preparación de los datos

## Conversión a numérico

Hasta aquí, todo luce bien, pero como ya lo mencionamos, XGBoost requiere matrices numéricas para funcionar correctamente, de modo que debemos convertir nuestras columnas de datos de tipo cadena de texto a tipo numérico.

Es aquí cuando, casi por reflejo, te estás preparando para realizar un one hot encoding o generación de variables dummy de nuestros datos. Después de todo, es lo que se *debe* hacer para convertir datos categóricos a numéricos ¿no es así?

Generalmente sí, pero no lo haremos para este ejemplo.

Una ventaja de XGBoost es que puede trabajar con datos categóricos que han sido codificados numéricamente. Esta **no es una característica de todos los algoritmos de clasificación**, por lo que es recomendable hacer one hot encoding o dummy variables siempre que sea posible.

Para este ejemplo, con fines demostrativos, haremos recodificación y veremos qué resultados obtenemos. Un buen ejercicio es que, por tu cuenta y con estos mismos datos, realices el entrenamiento del modelo que crearemos usando one hot encoding o dummy variables, y compares los resultados obtenidos.

Dicho esto, continuamos con la codificación.

Definiremos una función que realice esta conversión en nuestro data frame `agar`.

Para cada columna en el data frame:

* Obtenemos los valores únicos que se encuentran en ella con el método `unique()`, estos son los datos originales.
* Creamos una lista de reemplazos, a partir de la cantidad de valores únicos que hemos encontrado. El primer valor único será igual a 0, el segundo a 1 y así sucesivamente.
* Generamos un diccionario, que será nuestro mapa de reemplazo, con los valores originales y los reemplazos.
* Aplicamos el mapa de reemplazos con el método `replace()`

``` python
def str_a_num(df):
    for col in df:
        original = df[col].unique()
        reemplazo = list(range(len(original)))
        mapa = dict(zip(original, reemplazo))
        df[col] = df[col].replace(mapa)
    return(df)
```

Usamos nuestra función y el resultado es el siguiente.

``` python
agar = str_a_num(agar)
agar.head(5)
```
``` 
   target  cap_shape  cap_surface  ...  spore_print_color  population  habitat
0       0          0            0  ...                  0           0        0
1       1          0            0  ...                  1           1        1
2       1          1            0  ...                  1           1        2
3       0          0            1  ...                  0           0        0
4       1          0            0  ...                  1           2        1

[5 rows x 23 columns]
```

## Sets de entrenamiento y prueba

Como es el caso para todos los algoritmos de predicción supervisados, necesitamos dividir nuestros datos en un conjunto de entrenamiento, que será usado para aprender las características de los datos y generar un modelo de predicción; y un conjunto de prueba, con el que validamos el modelo generado.

Generamos nuestro set de entrenamiento con la función `train_test_split()` de `sklearn`. 

Establecemos el tamaño de nuestro set de prueba a 30% del total de datos con el argumento `test_size=.3`, de este modo, el set de entramiento tendrá un tamaño del 70% del total de datos. Finalmente, el argumento `random_state=1999` es usado para replicar los resultados.

Asignamos estos sets a los objetos agar_train y agar_test.

``` python
agar_train, agar_test = train_test_split(agar, test_size=.3, random_state=1999)
```

Veamos el tamaño de estos conjuntos de datos con el método `shape`.

``` python
agar_train.shape
```
``` python
(5686, 23)
```
``` python
agar_test.shape
```
``` python
(2438, 23)
```

## Convertir a DMatrix

Como ya lo mencionamos, la implementación XGBoost de R requiere que los datos que usemos sean matrices, específicamente de tipo *DMatrix*, así que necesitamos convertir nuestros sets de entrenamiento y prueba a este tipo de estructura.

Usaremos la función `xgb.DMatrix()` de `xgboost` para la conversión.

Esta función acepta como argumento `data` un arreglo de numpy, o un objeto con estructura similar como es el caso de los data frames de pandas.Se pueden especificar algunos atributos adicionales al objeto que devolverá, nosotros definiremos el atributo label para identificar la variable objetivo en nuestros datos.

Al usar esta función es **muy importante que tu argumento data no incluya la columna con la variable objetivo**, de lo contrario, obtendrás una precisión perfecta en tus predicciones, las cual será inútil con datos nuevos.

Entonces, quitamos la variable objetivo de los datos usando el método `drop()` y creamos dos matrices, una para nuestro set de entrenamiento y otra para el set de prueba.

``` python
agar_train_mat = xgb.DMatrix(agar_train.drop("target", 1), label=agar_train["target"])
agar_test_mat = xgb.DMatrix(agar_test.drop("target", 1), label=agar_test["target"])
```

Nuestro resultados deben ser similares a estos.

``` python
agar_train_mat
```
``` 
<xgboost.core.DMatrix object at 0x00000298F2DA38B0>
```

¡Listo! Hemos concluido la parte más laboriosa de todo el proceso. Podemos comenzar con el entrenamiento del modelo predictivo.

# Entrenamiento del modelo predictivo

Para entrenar un modelo usamos la función `train()` de `xgboost`.

Tenemos a nuestra disposición una amplia cantidad de hiper parámetros para ajustar, pero para este ejemplo introductorio haremos ajustes solo a los siguientes:

* `booster`: El tipo de modelo de clasificación usado, por defecto *gbtree*.
* `objective`: El tipo de tarea de clasificación que realizaremos. Para clasificación binaria, nuestro caso, especificamos *binary:logistic*.
* `max_depth`: “Profundidad” o número de nodos de bifurcación de los árboles de decisión usados en el entrenamiento. Aunque una mayor profundidad puede devolver mejores resultados, también puede resultar en overfitting (sobre ajuste).
* `eta`: La tasa de aprendizaje del modelo. Un mayor valor llega más rápidamente al mínimo de la función objetivo, es decir, a un “mejor modelo”, pero puede “pasarse” de su valor óptimo. En cambio, un valor pequeño puede nunca llegar al valor óptimo de la función objetivo, incluso después de muchas iteraciones. En ambos casos, esto afecta el desempeño de nuestros modelos con nuevos.
* `nthread`: El número de hilos computacionales que serán usados en el proceso de entrenamiento. Generalmente se refiere a los núcleos del procesador de tu equipo de cómputo, local o remoto, pero también pueden ser los núcleos de un GPU.
* `nround`: El número de iteraciones que se realizarán antes de detener el proceso de ajuste. Un mayor número de iteraciones generalmente devuelve mejores resultados de predicción, pero necesita más tiempo de entrenamiento y conlleva un riesgo de sobre ajuste si son demasiadas rondas.

Como los datos de nuestro ejemplo son sencillos, definiremos valores muy conservadores para todos estos hiper parámetros.

Guardamos los hiper parámetros en un diccionario llamado `parámetros` y el número de iteraciones en su propia variable `rondas`.
``` python
parametros = {"booster":"gbtree", "max_depth": 2, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
rondas = 10
```

Además, definimos una variable que contiene una lista de tuplas con los sets que serán usados para realizar la validación del modelo después de cada ronda. Este paso es opcional, pero es sumamente recomendable llevarlo a cabo, pues así monitoreamos el proceso de entrenamiento fácilmente.

``` python
evaluacion = [(agar_test_mat, "eval"), (agar_train_mat, "train")]
```

Con estas variables definidas, continuamos con el entrenamiento del modelo. Notarás que se irán mostrando los resultados de evaluación de cada iteración del modelo, hasta que alcance el número de rondas que hemos definido.

``` python
modelo = xgb.train(parametros, agar_test_mat, rondas, evaluacion)
```
``` 
[0]     eval-error:0.01600      train-error:0.01425
[1]     eval-error:0.01600      train-error:0.01425
[2]     eval-error:0.01600      train-error:0.01425
[3]     eval-error:0.01600      train-error:0.01425
[4]     eval-error:0.01600      train-error:0.01425
[5]     eval-error:0.01600      train-error:0.01425
[6]     eval-error:0.01600      train-error:0.01425
[7]     eval-error:0.01600      train-error:0.01425
[8]     eval-error:0.01600      train-error:0.01425
[9]     eval-error:0.01600      train-error:0.01425
```

Tu resultado debe verse como lo siguiente.

``` python
modelo
```
``` 
<xgboost.core.Booster object at 0x00000298F2DA3A60>
```

# Generación de predicciones

El siguiente paso es utilizar el método `predict()` de nuestro modelo con los datos de prueba para generar predicciones.

Para usar este método es muy importante que tus datos de prueba tengan la misma estructura que tus datos de entrenamiento, es decir, mismo número de variables o features, si tienen una estructura diferente a los de entrenamiento, no podrás obtener predicciones.

Generamos las predicciones y las asignamos al objeto `prediccion`.

``` python
prediccion = modelo.predict(agar_test_mat)
```

Nuestro resultado es un arreglo de valores numéricos, cada uno representando la probabilidad de que un caso en particular pertenezca al valor 1 de nuestra variable objetivo. Es decir, la probabilidad de que ese hongo sea venenoso.

``` 
prediccion

array([0.02051316, 0.96254647, 0.8165461 , ..., 0.02778091, 0.96254647,
       0.96254647], dtype=float32)
```

Para nuestro caso, tomaremos las probabilidades mayores a 0.5 como una predicción de pertenencia al valor 1 de nuestra variable objetivo. Es un umbral bastante relajado. 

En realidad, considerando que estamos tratando de clasificar hongos venenosos, debería ser más exigente, pero lo fijaremos de este modo para fines de demostración de las capacidades de XGBoost.

Hacemos la recodificación de probabilidad a valores binarios con el siguiente bloque de código y veamos los primeros diez elementos resultantes.

``` python
prediccion = [1 if i > .5 else 0 for i in prediccion]
prediccion[:10]
```
``` 
[0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
```

Ahora veamos qué tan acertadas han sido nuestras predicciones

# Evaluación del modelo

Para evaluar nuestro modelo comparamos nuestras predicciones contra las categorías reales de nuestro set de prueba.

Usaremos tres funciones de *sci kit learn*:

* `confusion_matrix()`: Genera una matriz de confusión, en la cual vemos nuestros verdaderos y falsos positivos, así como los verdaderos y falsos negativos.
* `accuracy_score()`: El valor de predicción que hemos obtenido. En nuestro caso, expresado como un valor entre 0 y 1.
* `classification_report()`: Métricas adicionales a la precisión. En particular útiles para identificar qué tanto éxito tuvimos clasificando los casos positivos en comparación contra los negativos.

Todas estas funciones requieren como argumentos los valores reales, que nosotros tenemos en la variable "target" del objeto `agar_test`, así como los valores de predicción que acabamos de calcular.

Para facilitar el uso de estas funciones, definimos una función que nos devuelve resultados de las tres.

``` python
def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)
```

Usamos nuestra función con nuestros datos reales y de predicción. Obtenemos los siguientes resultados:

``` python
metricas = metricas(agar_test["target"], prediccion)
[print(i) for i in metricas]

```
``` 
[[1175    0]
 [  67 1196]]
0.972518457752256
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      1175
           1       1.00      0.95      0.97      1263

    accuracy                           0.97      2438
   macro avg       0.97      0.97      0.97      2438
weighted avg       0.97      0.97      0.97      2438

```

Nada mal. Tuvimos una precisión del 97% y sobresale que tuvimos un valor predictivo de 100% para la clase positiva, un hongo venenoso.

En la práctica, con datos reales, rara vez obtenemos resultados tan buenos con tan poco esfuerzo, pero si comparas estos resultados contra los de árboles de decisión convencionales, notarás una gran diferencia en desempeño a favor de XGBoost.

Después de preparar nuestros datos, la tarea que más tiempo consume al usar este modelo es encontrar los mejores hiper parámetros para alcanzar la mayor precisión posible de un modelo.

Veamos qué pasa si ajustamos nuestros hiper parámetros con un segundo modelo.

## Segundo modelo.

Este segundo modelo tiene un número de iteraciones mayor que el anterior, 100 en lugar de 10, y una mayor profundidad en los árboles generados, 4 en lugar de 2.

Además, agregamos el hiper parámetro `early_stopping_rounds=10` a la función `train()`, para que el entrenamiento se detenga si después de diez iteraciones consecutivas no hay mejora en el modelo. Este hiper parámetro es sumamente útil para acortar el tiempo de entrenamiento de un modelo, pues evita que el proceso continúe si ya no se obtienen mejores resultados de predicción.

Las métricas de evaluación son las mismas, así que usamos la misma variable que ya hemos definido.

``` python
parametros_02 = {"booster":"gbtree", "max_depth": 4, "eta": .3, "objective": "binary:logistic", "nthread":2}
rondas_02 = 100

modelo_02 = xgb.train(parametros_02, agar_test_mat, rondas_02, evaluacion, early_stopping_rounds=10)
```

```
[0]     eval-error:0.00697      train-error:0.00545
Multiple eval metrics have been passed: 'train-error' will be used for early stopping.       

Will train until train-error hasn't improved in 10 rounds.
[1]     eval-error:0.00697      train-error:0.00545
[2]     eval-error:0.00328      train-error:0.00281
[3]     eval-error:0.00082      train-error:0.00105
[4]     eval-error:0.00082      train-error:0.00105
[5]     eval-error:0.00082      train-error:0.00105
[6]     eval-error:0.00082      train-error:0.00105
[7]     eval-error:0.00082      train-error:0.00105
[8]     eval-error:0.00082      train-error:0.00105
[9]     eval-error:0.00082      train-error:0.00105
[10]    eval-error:0.00082      train-error:0.00105
[11]    eval-error:0.00082      train-error:0.00105
[12]    eval-error:0.00082      train-error:0.00105
[13]    eval-error:0.00082      train-error:0.00105
Stopping. Best iteration:
[3]     eval-error:0.00082      train-error:0.00105
```

Podemos ver que este modelo se ha detenido antes que el primero y nos da como mejor iteración la tercera de trece que tuvo en total.

Veamos ahora sus métricas de evaluación.

``` python
prediccion_02 = modelo_02.predict(agar_test_mat)
prediccion_02 = [1 if i > .5 else 0 for i in prediccion_02]

metricas_02 = metricas(agar_test["target"], prediccion_02)
[print(i) for i in metricas_02]
```

``` 
[[1162    2]
 [   0 1274]]
0.9991796554552912
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1164
           1       1.00      1.00      1.00      1274

    accuracy                           1.00      2438
   macro avg       1.00      1.00      1.00      2438
weighted avg       1.00      1.00      1.00      2438

```

Este segundo modelo tiene precisión del 100%, tanto para la clase positiva como para la negativa. Nada mal, aunque vale la pena reiterar que generalmente no obtenemos resultados tan buenos y con tan poco esfuerzo en la práctica cotidiana. 

Así que no te desanimes si no obtienes precisiones perfectas en tus propios proyectos, aun después de haber afinado muchas veces tus modelos. Es normal, obtener buenos resultados de predicción no es una tarea fácil.

# Exporta e importa modelos

Exportar un modelo a un archivo es una tarea clave para compartirlo con otras personas y poder implementarlos en producción.

Exportar modelos es bastante sencillo con XGBoost, solo necesitamos el método `save_model()`. Por supuesto, es muy importante tener cuidado con la versión de XGBoost usada para generar el modelo, para así evitar problemas de compatibilidad.

Como ejemplo, exportaremos nuestro `modelo_02`, que nos ha dado buenos resultados.

``` python
modelo_02.save_model("modelo_02.model")
```

Una vez que hemos exportado un podemos importarlo con el método `load_model()`. Para ello, primero creamos un modelo vacío con la función `Booster()` y entonces importamos nuestro modelo a este objeto.

``` python
modelo_importado = xgb.Booster()
modelo_importado.load_model("modelo_02.model")
```

Con el modelo importado podemos realizar predicciones. Si ejecutas el siguiente bloque de código obtendrás los mismos resultados que con el objeto `modelo_02`.

``` python
modelo_importado.predict(agar_test_mat)
```

Una gran ventaja de XGBoost es que un modelo exportado es compatible con las distintas implementaciones del algoritmo. Es decir, puedes importar un modelo que has entrenado usando Python usando R y viceversa. 

Esta es una característica muy valiosa para facilitar la colaboración y que te da flexibilidad en el uso de herramientas durante un proyecto de data science.

# Para concluir

En este artículo hemos revisado, de manera general, la implementación para Python del algoritmo XGBoost. En el proceso, también dimos un vistazo al proceso para preparar datos con formatos no convencionales para ser usados en este algoritmo y cómo podemos exportar un modelo de predicción generado con XGBoost una vez que lo hemos entrenado.

Esta revisión no ha sido exhaustiva, hay algunos temas que es importante estudiar para obtener mejores resultados al usar XGBoost:

* En nuestro ejemplo, tomamos los datos tal cual los obtuvimos. En la práctica, es **esencial** realizar una exploración mucho más profunda de los datos, por ejemplo, analizar las correlaciones entre variables, el comportamiento de datos perdidos e identificar desequilibrio en la variable objetivo. Rara vez tendrás datos que tienen desde un principio el formato ideal para trabajar con ellos.
* Los hiper parámetros que usamos en nuestro ejemplo no son los únicos que tiene XGBoost, además de que existen más boosters. Conocerlos todos y entender **cómo funcionan** y esto como afecta al comportamiento del algoritmo te da la capacidad para hacer ajustes razonados y obtener mejores resultados de predicción.
* Sólo realizamos una tarea de clasificación binaria, pero no es la única que puede revisar XGBoost. Cada una de ellas requiere de una preparación de datos y ajuste de hiper parámetros diferente.

Si quieres conocer más sobre estos temas un buen punto de partida es la documentación de XGBoost.

* [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/index.html)

---

Consultas, dudas, comentarios y correcciones son bienvenidas:

* jboscomendoza@gmail.com

El código y los datos usados en este documento se encuentran en Github:

* https://github.com/jboscomendoza/xgboost_en_python

También puedes encontrar cómo usar XGBoost en R en este enlace:

* [XGBoost en R](https://medium.com/@jboscomendoza/xgboost-en-r-398e7c84998e)