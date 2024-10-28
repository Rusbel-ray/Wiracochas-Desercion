from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    data = request.form
    features = np.array([
        float(data['edad']),
        int(data['genero']),
        float(data['promedio_calificaciones_semestre_anterior']),
        int(data['asignaturas_reprobadas']),
        int(data['asignaturas_aprobadas']),
        int(data['numero_semestres_matriculado']),
        float(data['asistencia']),
        float(data['distancia_universidad']),
        int(data['trabaja']),
        int(data['frecuencia_consumo_alcohol']),
        int(data['consumo_tabaco']),
        int(data['bica']),
        int(data['trica']),
    ]).reshape(1, -1)

    # Escalar características
    features_scaled = scaler.transform(features)

    # Hacer predicción
    prediction = model.predict(features_scaled)

    # Devolver el resultado
    if prediction[0] == 1:
        result = "El modelo predice que el estudiante desertará."
    else:
        result = "El modelo predice que el estudiante no desertará."

    return render_template('index.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
