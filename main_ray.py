import pathlib
import time

import numpy as np
import ollama
import ray

# Modèles identiques dans main.py et main_ray.py pour une comparaison juste
DATASET_PATH = "HarryPotter.txt"
EMBEDDING_MODEL = "embeddinggemma:300m"
LANGUAGE_MODEL = "gemma3:270m"
TOP_N = 3


# -------------------------
# 1. Initialisation
# -------------------------

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

def retrieve(query, vector_db, top_n=TOP_N):

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

def main(ray_init_time=0.0):

    # -------------------------
    # 6. Load dataset
    # -------------------------

    t0 = time.perf_counter()
    dataset = create_dataset(DATASET_PATH)
    t1 = time.perf_counter()

    # -------------------------
    # 7. Split dataset into chunks
    # -------------------------

    chunks = [
        line.strip()
        for line in dataset.splitlines()
        if line.strip()
    ]
    t2 = time.perf_counter()

    # -------------------------
    # 8. Generate embeddings with Ray
    # -------------------------

    print("Generating embeddings with Ray...")

    vector_db = build_vector_db(chunks)

    t3 = time.perf_counter()

    print(f"Vector database created: {len(vector_db)} chunks")

    # -------------------------
    # 9. User query
    # -------------------------

    input_query = input("\nAsk me a question: ")

    # -------------------------
    # 10. Retrieve relevant chunks
    # -------------------------

    t_retrieval_start = time.perf_counter()
    retrieved_knowledge = retrieve(
        input_query,
        vector_db,
        top_n=TOP_N
    )
    t4 = time.perf_counter()

    print("\nRetrieved knowledge:")

    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")

    # -------------------------
    # 11. Build prompt
    # -------------------------

    context = "\n".join(
        [f" - {chunk}" for chunk, similarity in retrieved_knowledge]
    )

    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question.
If the answer is not present, say that you don't know.

Context:
{context}
"""

    # -------------------------
    # 12. Generate response
    # -------------------------

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

    t5 = time.perf_counter()

    # -------------------------
    # 13. Execution time (hors temps de saisie)
    # -------------------------

    retrieval_time = t4 - t_retrieval_start
    total_time = (
        ray_init_time
        + (t1 - t0)
        + (t2 - t1)
        + (t3 - t2)
        + retrieval_time
        + (t5 - t4)
    )

    print("\n\n========== EXECUTION (RAY) ==========")
    print(f"Ray init:              {ray_init_time:.2f}s")
    print(f"Dataset loading:       {t1 - t0:.2f}s")
    print(f"Chunking:              {t2 - t1:.2f}s")
    print(f"Embedding generation:  {t3 - t2:.2f}s")
    print(f"Retrieval:             {retrieval_time:.2f}s")
    print(f"LLM generation:        {t5 - t4:.2f}s")
    print("--------------------------------------")
    print(f"Total (hors saisie):   {total_time:.2f}s")


if __name__ == "__main__":
    ray_start = time.perf_counter()
    ray.init()
    ray_init_time = time.perf_counter() - ray_start

    try:
        main(ray_init_time)
    finally:
        ray.shutdown()
