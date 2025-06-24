from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Documentos con información sobre animales.
documentos = [
    "El veloz zorro marrón salta sobre el perro perezoso.",
    "Un perro marrón persiguió al zorro.",
    "El perro es perezoso.",
]

#Convertir documentos a vectores usando TF-IDF.
vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(documentos)

#Calcular la similitud del coseno entre los documentos.
similitud_coseno = cosine_similarity(matriz_tfidf, matriz_tfidf)

#Imprimir los resultados
print("Matriz de Similitud del Coseno:")
print(similitud_coseno)