# Clasificación: Primeros Pasos

Este proyecto se centra en la aplicación de técnicas de **machine learning** para predecir la adhesión a inversiones de clientes de un banco, basándose en datos de una campaña de marketing.

## Objetivo del Proyecto

El objetivo principal es desarrollar un modelo de clasificación capaz de predecir si un cliente invertirá o no, utilizando un enfoque paso a paso que incluye lectura, análisis exploratorio, separación y transformación de datos, ajuste, evaluación y comparación de modelos.

## Estructura del Proyecto

El notebook está dividido en las siguientes secciones principales:

### 1. Análisis Exploratorio de Datos (EDA)

*   **Lectura de Datos**: Se cargan los datos desde el archivo `marketing_inversiones.csv` utilizando la librería `pandas`.
*   **Verificación de Calidad**: Se utiliza `info()` para verificar tipos de datos y la presencia de valores nulos.
*   **Exploración Visual**: Se emplean gráficos de `plotly.express` para visualizar la distribución de variables categóricas (como `estado_civil`, `escolaridad`, `default`, `prestatario`) y numéricas (como `edad`, `saldo`, `ultimo_contacto`, `ct_contactos`) en relación con la variable objetivo `adherencia_inversion`.

### 2. Transformación de los Datos

*   **Separación de Variables**: Se distinguen las variables explicativas (`X`) de la variable de respuesta (`y`).
*   **Transformación de Variables Explicativas**: Las variables categóricas en `X` se transforman a formato numérico utilizando `OneHotEncoder` de `sklearn.compose` para que sean interpretables por los algoritmos de machine learning.
*   **Transformación de la Variable Respuesta**: La variable `adherencia_inversion` en `y` se convierte a un formato numérico (0 o 1) utilizando `LabelEncoder`.

### 3. Ajuste de Modelos

*   **División de Datos**: Se dividen los datos en conjuntos de entrenamiento y prueba (`X_train`, `X_test`, `y_train`, `y_test`) utilizando `train_test_split` para evaluar el rendimiento del modelo en datos no vistos.
*   **Modelo de Referencia (Baseline)**: Se entrena un `DummyClassifier` como modelo base para comparar el rendimiento de otros modelos.
*   **Árboles de Decisión**: Se implementa y entrena un `DecisionTreeClassifier`. Se explora su rendimiento con y sin `max_depth` limitado para controlar el sobreajuste.
    *   Se incluye una visualización del árbol de decisión para facilitar la interpretabilidad.
*   **Normalización de Datos**: Se normalizan las variables numéricas utilizando `MinMaxScaler` para evitar que la escala de los valores influya incorrectamente en ciertos algoritmos.
*   **KNN (K-Nearest Neighbors)**: Se entrena un `KNeighborsClassifier` utilizando los datos normalizados.

### 4. Selección y Serialización de Modelos

*   **Comparación de Modelos**: Se evalúa y compara la exactitud (`score`) de los modelos `DummyClassifier`, `DecisionTreeClassifier` y `KNeighborsClassifier` en el conjunto de prueba.
*   **Serialización**: El modelo `OneHotEncoder` y el `DecisionTreeClassifier` con mejor rendimiento (`modelo_arbol`) se serializan utilizando `pickle` para su uso futuro en producción.

## Cómo Utilizar el Modelo Serializado

Para realizar predicciones con un nuevo dato, puedes seguir los siguientes pasos:

1.  **Cargar los modelos**: Carga los archivos `.pkl` guardados (el `OneHotEncoder` y el modelo 'champion').
2.  **Preparar el nuevo dato**: Crea un DataFrame de pandas con el nuevo dato, asegurándote de que tenga las mismas columnas que el conjunto de datos original.
3.  **Transformar el nuevo dato**: Aplica las mismas transformaciones (One-Hot Encoding, por ejemplo) al nuevo dato que se aplicaron a los datos de entrenamiento.
4.  **Realizar la predicción**: Usa el modelo cargado para predecir la clase del nuevo dato.

```python
import pandas as pd
import pickle

# Crear un nuevo dato de ejemplo
nuevo_dato = {
    'edad': [45],
    'estado_civil':['soltero (a)'],
    'escolaridad':['superior'],
    'default': ['no'],
    'saldo': [23040],
    'prestatario': ['no'],
    'ultimo_contacto': [800],
    'ct_contactos': [4]
}
nuevo_dato_df = pd.DataFrame(nuevo_dato)

# Cargar los modelos serializados
modelo_one_hot = pd.read_pickle('/content/modelo_onehotencoder.pkl')
modelo_arbol = pd.read_pickle('/content/modelo_champion.pkl')

# Transformar el nuevo dato usando el OneHotEncoder cargado
nuevo_dato_transformado = modelo_one_hot.transform(nuevo_dato_df)

# Realizar la predicción
prediccion = modelo_arbol.predict(nuevo_dato_transformado)

# Mostrar el resultado
if prediccion[0] == 1:
    print("El modelo predice que el cliente SÍ se adherirá a la inversión.")
else:
    print("El modelo predice que el cliente NO se adherirá a la inversión.")

```

## Tecnologías Utilizadas

*   Python 3
*   Pandas
*   Scikit-learn
*   Plotly
