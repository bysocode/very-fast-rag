
import numpy as np
import pathlib
import ollama
import ray
import time

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []


# -------------------------
# 1. Initialisation
# -------------------------
start = time.start()
def create_dataset(filepath):
    path = pathlib.Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    return path.read_text(encoding="utf-8")


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def get_embedding(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    # Compatible avec les réponses dict et objets d'Ollama
    if isinstance(response, dict):
        return response["embeddings"][0]

    return response.embeddings[0]


# -------------------------
# 2. Fonctions Ray
# -------------------------

@ray.remote
def embed_chunk(chunk):
    """Génère l'embedding d'un chunk."""
    embedding = get_embedding(chunk)
    return chunk, embedding


@ray.remote
def embed_query(query):
    """Génère l'embedding de la requête."""
    return get_embedding(query)


@ray.remote
def score_chunk(query_embedding, chunk, embedding):
    """Calcule la similarité entre la requête et un chunk."""
    similarity = cosine_similarity(query_embedding, embedding)
    return chunk, similarity


# -------------------------
# 3. Création de la base vectorielle
# -------------------------

def build_vector_db(chunks):
    # Distribution des calculs d'embeddings
    futures = [
        embed_chunk.remote(chunk)
        for chunk in chunks
    ]

    # Récupération des résultats
    return ray.get(futures)


# -------------------------
# 4. Retrieval avec Ray
# -------------------------

def retrieve(query, vector_db, top_n=3):

    # Embedding de la requête
    query_embedding = ray.get(
        embed_query.remote(query)
    )

    # Calcul parallèle des similarités
    futures = [
        score_chunk.remote(query_embedding, chunk, embedding)
        for chunk, embedding in vector_db
    ]

    similarities = ray.get(futures)

    # Tri par ordre décroissant
    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_n]


# -------------------------
# 5. Application principale
# -------------------------

def main():

    # Chargement du dataset
    dataset = create_dataset("data.txt")

    chunks = [
        line.strip()
        for line in dataset.splitlines()
        if line.strip()
    ]

    # Construction de la base vectorielle
    print("Generating embeddings with Ray...")

    vector_db = build_vector_db(chunks)

    print(f"Vector database created: {len(vector_db)} chunks")

    # Question utilisateur
    input_query = input("\nAsk me a question: ")

    # Recherche des chunks pertinents
    retrieved_knowledge = retrieve(
        input_query,
        vector_db,
        top_n=3
    )

    print("\nRetrieved knowledge:")

    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")

    # Construction du contexte
    context = "\n".join(
        [f" - {chunk}" for chunk, similarity in retrieved_knowledge]
    )

    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question.
If the answer is not present, say that you don't know.

Context:
{context}
"""

    # Génération de la réponse avec Ollama
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    print("\nChatbot response:")

    for chunk in stream:
        print(
            chunk["message"]["content"],
            end="",
            flush=True
        )



if __name__ == "__main__":

    ray.init()

    start = time.perf_counter()

    try:
        main()

    finally:
        end = time.perf_counter()
        ray.shutdown()

        print(f"\nExécuté en {end - start:.2f} secondes")

