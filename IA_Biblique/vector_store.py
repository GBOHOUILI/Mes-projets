'''Connexion à Huggingface'''

from huggingface_hub import notebook_login
notebook_login()
# BAAI/bge-m3 should be a separate line, likely used in a model loading context.
# For example:
# from transformers import AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/bge-m3")

'''Embeddings'''

from qdrant_client import models, QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import util
import numpy as np

# Initialiser le client Qdrant
client = QdrantClient(":memory:")  # Stockage en mémoire, remplacez par la persistance si nécessaire

# Utiliser le modèle "BAAI/bge-m3" pour les embeddings
encoder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Fonction pour stocker les vecteurs dans Qdrant
def vector_store1(data, encoder):
    # Créer une collection Qdrant avec la distance COSINE et la taille des vecteurs
    client.create_collection(
        collection_name="Ia_Biblique",
        vectors_config=models.VectorParams(
            distance=models.Distance.COSINE,
            size=encoder.model.get_sentence_embedding_dimension()  # Taille des embeddings
        ),
    )

    points = []
    point_id = 1
    for _, row in data.iterrows():
        # Encoder chaque chunk de texte
        for chunk in row['text_chunks']:  # Encodage de chaque partie du verset
            vector = encoder.embed_query(chunk)
            payload = {
                'Book Name': row['Book Name'],  # Nom du livre
                'Chapter': row['Chapter'],  # Chapitre
                'Verse': row['Verse'],  # Verset
                'Text': row['Text']  # Texte original complet
            }
            points.append(
                models.PointStruct(
                    id=point_id,  # ID unique pour chaque point
                    vector=vector,  # Embedding du chunk
                    payload=payload  # Metadonnées associées
                )
            )
            point_id += 1  # Incrémenter l'ID pour chaque point

    # Upload des points dans la collection
    client.upload_points(
        collection_name="Ia_Biblique",
        points=points,
    )

# Fonction pour rechercher des points similaires avec une question donnée
def search_similarity(question):
    hits = client.search(
        collection_name="Ia_Biblique",
        query_vector=encoder.embed_query(question),  # Encodage de la question
        limit=5,  # Limite des résultats à 5, ajustez si nécessaire
    )
    return hits

# Fonction pour récupérer les résultats sous forme de versets
def retriever(question):
    hits = search_similarity(question)
    responses = [hit.payload for hit in hits]  # Retourner les metadonnées associées aux versets trouvés
    return responses

# Fonction pour calculer la similarité cosinus entre les questions et les réponses
def cosine_similarity(questions, answers):
    similarities = []
    for question, answer in zip(questions, answers):
        results = search_similarity(question)
        for result in results:
            similarity = util.cos_sim(encoder.embed_query(result.payload['Text']), encoder.embed_query(answer))
            similarities.append(similarity)
    return np.mean(similarities)

# Fonction pour calculer la distance euclidienne entre deux embeddings
def euclidean_distance(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance




'''from sentence_transformers import SentenceTransformer

# Remplacez l'encodeur HuggingFaceEmbeddings par SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-m3")

def vector_store(data, encoder):
    # Créer une collection dans Qdrant
    client.create_collection(
        collection_name="IA_Biblique",
        vectors_config=models.VectorParams(
            distance=models.Distance.COSINE,
            size=encoder.get_sentence_embedding_dimension()  # Taille des embeddings
        ),
    )

    points = []
    point_id = 1
    for _, row in data.iterrows():
        for chunk in row['text_chunks']:  
            vector = encoder.encode(chunk)  # Encodez chaque chunk
            payload = {
                'Book Name': row['Book Name'],
                'Chapter': row['Chapter'],
                'Verse': row['Verse'],
                'Text': row['Text']
            }
            points.append(
                models.PointStruct(
                    id=point_id,  # ID unique pour chaque point
                    vector=vector,
                    payload=payload
                )
            )
            point_id += 1  # Incrémenter l'ID pour chaque point

    # Téléverser les points dans la collection
    client.upload_points(
        collection_name="IA_Biblique",
        points=points,
    )
'''

vector_store(data, encoder)