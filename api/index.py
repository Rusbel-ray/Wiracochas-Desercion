from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Carga el modelo
model = joblib.load('model/random_forest_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Aquí procesas los datos del formulario y predices el resultado usando el modelo
        data = request.form
        # Suponiendo que tienes los datos listos en 'X_new' para predecir
        X_new = [[int(data['edad']), int(data['genero']), ...]]  # Ajusta con las características de entrada de tu modelo
        prediction = model.predict(X_new)
        
        return render_template('form.html', prediction=prediction)
    return render_template('form.html')

# Para que Vercel lo detecte como una función
def handler(request, context):
    return app(request.environ, start_response=context.start_response)