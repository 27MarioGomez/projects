# Crypto Price Predictions

Crypto Price Predictions es un dashboard interactivo desarrollado en Streamlit para predecir el precio futuro de criptomonedas. El sistema integra datos históricos obtenidos de yfinance, indicadores técnicos y análisis de sentimiento (noticias y Fear & Greed) para generar predicciones basadas en un ensamble de modelos que incluyen LSTM, XGBoost y Prophet.

## Descripción del Dashboard

El dashboard realiza las siguientes tareas:

- **Extracción de Datos:**  
  Descarga datos históricos de criptomonedas mediante yfinance. Los datos se obtienen dinámicamente a través de APIs, por lo que no se incluye un directorio de datos en el repositorio.

- **Cálculo de Indicadores Técnicos:**  
  Se calculan indicadores esenciales como RSI, MACD, Bollinger Bands, SMA, ATR, OBV, EMA200, log_return y vol_30d, además del análisis de sentimiento.

- **Optimización de Features:**  
  Se utiliza un pipeline que aplica imputación (mediana), selección automática de features (usando ElasticNetCV refinado con XGBoost) y escalado, utilizando únicamente los datos del último año para el entrenamiento sin afectar la visualización completa.

- **Análisis de Sentimiento:**  
  Se combina el sentimiento derivado de noticias (usando NewsApiClient y modelos de análisis de sentimiento) con el índice Fear & Greed para ajustar las predicciones.

- **Modelos de Predicción:**  
  Se implementa un ensamble de:
  - **LSTM:** Con hiperparámetros optimizados mediante Keras Tuner (Hyperband) en 8 épocas.
  - **XGBoost:** Para predicción iterativa a corto plazo.
  - **Prophet:** Para predicciones a mediano/largo plazo, anclando el primer valor al precio actual.

> **NFA:** Not Financial Advice

## Acceso al Dashboard

El dashboard está desplegado en la nube y se puede acceder desde la siguiente URL:  
[https://cryptopricepredictions.streamlit.app/](https://cryptopricepredictions.streamlit.app/)

## Estructura del Proyecto

```
CRYPTO_PREDICTION/
├── README.md               # Documentación del proyecto (este archivo).
├── main.py                 # Código completo de la aplicación.
├── requirements.txt        # Dependencias necesarias para desplegar el dashboard en Streamlit.
```


## Notas Adicionales

- **Datos y Modelos:**  
  Los datos se extraen dinámicamente de diversas APIs, por lo que no se almacena un archivo de datos en el repositorio. Asimismo, el modelo entrenado no se guarda de forma fija ya que su efectividad varía según la criptomoneda; por ejemplo, un modelo adecuado para Bitcoin podría no ser óptimo para criptomonedas de mayor volatilidad.

- **Código y Funcionamiento:**  
  El código implementa desde la descarga de datos, el cálculo de indicadores técnicos (con paralelización y vectorización) y análisis de sentimiento, hasta el entrenamiento y predicción con múltiples modelos. Además, se incorporan técnicas de optimización de hiperparámetros y ajuste iterativo de las predicciones.


