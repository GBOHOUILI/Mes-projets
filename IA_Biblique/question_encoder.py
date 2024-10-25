# Fonction pour encoder et rechercher la question dans Qdrant
def rechercher_biblique(question, encoder, client, collection_name="IA_Biblique", limit=5):
    """
    Encode la question et effectue une recherche dans Qdrant.
    
    :param question: La question saisie par l'utilisateur.
    :param encoder: L'encodeur utilisé pour générer l'embedding de la question.
    :param client: Le client Qdrant pour effectuer la recherche.
    :param collection_name: Le nom de la collection dans Qdrant (par défaut: 'IA_Biblique').
    :param limit: Le nombre maximum de résultats à retourner (par défaut: 5).
    :return: Les résultats de la recherche.
    """
    # Encoder la question
    question_embedding = encoder.encode(question).tolist()

    # Effectuer la recherche dans Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=limit
    )

    # Retourner les résultats
    return search_result

# Exemple d'utilisation de la fonction
question = "que dit la bible de l'amour ?"

# Appel de la fonction pour rechercher dans la collection 'IA_Biblique'
resultats = rechercher_biblique(question, encoder, client)

# Afficher les résultats
for result in resultats:
    print(result)
