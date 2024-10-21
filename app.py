from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Carga el modelo entrenado
model = joblib.load('model/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Obtén los datos del formulario
            edad = int(request.form['edad'])
            genero = int(request.form['genero'])  # 0 para masculino, 1 para femenino
            promedio = float(request.form['promedio'])
            reprobadas = int(request.form['reprobadas'])
            aprobadas = int(request.form['aprobadas'])
            semestres = int(request.form['semestres'])
            asistencia = float(request.form['asistencia'])
            distancia = float(request.form['distancia'])
            trabaja = int(request.form['trabaja'])  # 0 para no, 1 para sí
            alcohol = int(request.form['alcohol'])  # Frecuencia de consumo de alcohol
            tabaco = int(request.form['tabaco'])    # Consumo de tabaco

            # Crea un array con los datos ingresados
            data = np.array([[edad, genero, promedio, reprobadas, aprobadas, semestres,
                              asistencia, distancia, trabaja, alcohol, tabaco]])

            # Realiza la predicción
            prediction = model.predict(data)

            # Interpreta el resultado
            if prediction == 1:
                resultado = 'Posible deserción'
            else:
                resultado = 'No probable deserción'
                
            return render_template('form.html', prediction_text=f'Resultado: {resultado}')
        
        except Exception as e:
            return render_template('form.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
