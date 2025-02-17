from fastapi import FastAPI
from Models import ComentarioDTO
import joblib  # Para cargar el modelo entrenado

model = joblib.load("sentiment_model_es.pkl")

app = FastAPI(title="Clasificador de comentarios")

# Endpoint para clasificar el texto
@app.post("/clasificar")
def clasificar_texto(data: ComentarioDTO):
    """
    Endpoint que recibe un texto en español y devuelve si el sentimiento es positivo o negativo.
    """
    # Preprocesar el texto y hacer la predicción
    prediction = model.predict([data.texto])
    sentiment = "positivo" if prediction[0] == 1 else "negativo"
    return {"texto": data.texto, "sentimiento": sentiment}

# Ejecutar la API (usar: uvicorn nombre_del_archivo:app --reload)