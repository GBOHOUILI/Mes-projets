'''Chargement de la data'''

data = pd.read_csv('bible_cleaned.csv')
data.head()

'''Netoyage de la donnée'''


# Télécharger les stopwords en français
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

 #fonction pour nettoyer les caractères spéciaux
def nettoyer_caractere(texte):
    texte = texte.lower() #conversion en minuscules
    #texte = re.sub(r'[^a-zA-Zéèêàâîôùûç\s]', '',texte) #supprimer les caractères spéciaux
    mots = texte.split() #diviser le corpus(division par espace)
      #mots = [mot for mot in mots if mot not in stop_words] #supprimer les mots courants
    return ' '.join(mots) #mette les mots en chaîne de caractères



# Initialisation du RecursiveTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)


# Fonction pour appliquer le splitter sur chaque ligne de context
def split_text(text):
    return text_splitter.split_text(text) # Use split_text instead of split

# Application du text splitter sur chaque ligne de context
data['text_chunks'] = data['Text'].apply(lambda x: split_text(x))

# Affichage des résultats
print(data['Text'].iloc[0])
print(data['text_chunks'].iloc[0])