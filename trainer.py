# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib  # Para guardar el modelo

# Cargar el dataset
# Asegúrate de que el archivo CSV esté en la misma carpeta o proporciona la ruta correcta
df = pd.read_csv("imdb_reviews.csv")

# Verificar las columnas del dataset
print(df.columns)

# Preprocesamiento de datos
# Asumimos que las columnas son "review_es" (texto en español) y "sentimiento" (positivo/negativo)
X = df["review_es"]  # Texto de las reseñas
y = df["sentimiento"].apply(lambda x: 1 if x == "positivo" else 0)  # Convertir a 1 (positivo) y 0 (negativo)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline con TF-IDF y Naive Bayes
# TF-IDF convierte el texto en vectores numéricos, y Naive Bayes es un clasificador simple y eficaz
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred)}")

# Guardar el modelo entrenado para usarlo en la API
joblib.dump(model, "sentiment_model_es.pkl")