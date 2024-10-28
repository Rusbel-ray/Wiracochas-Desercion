# Predicción de Deserción Estudiantil

Este proyecto es una aplicación web que permite predecir la probabilidad de deserción estudiantil utilizando un modelo de Machine Learning basado en **Random Forest**. La aplicación permite al usuario ingresar características específicas de un estudiante y, mediante un modelo entrenado, devuelve una predicción sobre si el estudiante desertará o no.

## Descripción

La aplicación fue desarrollada utilizando **Flask** para el backend y HTML/CSS para el frontend. Las características de los estudiantes, tales como edad, género, promedio de calificaciones, asignaturas reprobadas/aprobadas, etc., se ingresan a través de un formulario web y se procesan para generar una predicción.

El modelo de predicción fue entrenado con un dataset de 1500 estudiantes utilizando **Random Forest**, que es ideal para este tipo de clasificación binaria.

## Características

- Predecir la deserción estudiantil con base en características ingresadas por el usuario.
- Interfaz amigable y visualmente atractiva, con estilos personalizados.
- Implementación en un servidor Flask para predicción en tiempo real.
- Desplegable en **Render** o en local para desarrollo.

## Requisitos

Para ejecutar este proyecto localmente, necesitas tener instalados:

- Python 3.7 o superior
- Flask
- scikit-learn
- joblib

Puedes instalar las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt




