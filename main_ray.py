import os
import pathlib
import time

# Ray détecte « uv run » et recrée un venv pour les workers. Comme VIRTUAL_ENV
# pointe vers le .venv du projet et que Ray empaquette un autre .venv, uv émet
# un warning et duplique l'environnement. On désactive cette propagation :
# les workers utilisent directement le Python du venv courant.
# Doit être défini AVANT l'import de ray (lu au chargement de ray_constants).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import numpy as np  # noqa: E402
import ollama  # noqa: E402
import ray  # noqa: E402

# Modèles identiques dans main.py et main_ray.py pour une comparaison juste
DATASET_PATH = "HarryPotter.txt"
EMBEDDING_MODEL = "embeddinggemma:300m"
LANGUAGE_MODEL = "gemma3:270m"
TOP_N = 3

# Un appel Ollama pour plusieurs chunks : moins d'overhead HTTP
EMBED_BATCH_SIZE = 32

# Constat mesuré : Ollama lance llama-server avec `-np 1` pour les modèles
# d'embedding, même avec OLLAMA_NUM_PARALLEL=4. Un seul serveur Ollama
# sérialise donc les embeddings et Ray n'apporte pas de gain dessus.
# Ray deviendrait utile avec plusieurs serveurs/backends d'embedding.

# Permet un test rapide : MAX_CHUNKS=512 uv run main_ray.py
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "0"))  # 0 = tout le dataset


# -------------------------
# 1. Initialisation
# -------------------------

def create_dataset(filepath):
    path = pathlib.Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    return path.read_text(encoding="utf-8")


def warmup():
    """Charge les modèles en mémoire pour ne pas les compter dans les timings."""
    print("Warming up models...")

    ollama.embed(model=EMBEDDING_MODEL, input=["warmup"])

    ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        options={"num_predict": 1},
        stream=False,
    )


def get_embeddings(texts):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    # Compatible avec les réponses dict et objets d'Ollama
    if isinstance(response, dict):
        return response["embeddings"]

    return response.embeddings


# -------------------------
# 2. Fonctions Ray
# -------------------------

@ray.remote
def embed_batch(batch):
    """Génère les embeddings d'un lot de chunks."""
    embeddings = get_embeddings(batch)
    return list(zip(batch, embeddings))


# -------------------------
# 3. Création de la base vectorielle
# -------------------------

def build_vector_db(chunks):
    batches = [
        chunks[start:start + EMBED_BATCH_SIZE]
        for start in range(0, len(chunks), EMBED_BATCH_SIZE)
    ]

    # Distribution des calculs d'embeddings
    futures = [
        embed_batch.remote(batch)
        for batch in batches
    ]

    # Récupération et mise à plat des résultats
    results = ray.get(futures)

    return [
        item
        for batch_result in results
        for item in batch_result
    ]


# -------------------------
# 4. Retrieval vectorisé (NumPy)
# -------------------------

def retrieve(query, vector_db, top_n=TOP_N):
    query_embedding = np.asarray(
        get_embeddings([query])[0],
        dtype=np.float32,
    )

    matrix = np.asarray(
        [embedding for _, embedding in vector_db],
        dtype=np.float32,
    )

    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    denominator = norms * query_norm

    similarities = np.divide(
        matrix @ query_embedding,
        denominator,
        out=np.zeros_like(norms),
        where=denominator != 0,
    )

    indices = np.argsort(similarities)[-top_n:][::-1]

    return [
        (vector_db[i][0], float(similarities[i]))
        for i in indices
    ]


# -------------------------
# 5. Application principale
# -------------------------

def main(ray_init_time=0.0):

    # -------------------------
    # 6. Warm-up (hors timings)
    # -------------------------

    warmup()

    # -------------------------
    # 7. Load dataset
    # -------------------------

    t0 = time.perf_counter()
    dataset = create_dataset(DATASET_PATH)
    t1 = time.perf_counter()

    # -------------------------
    # 8. Split dataset into chunks
    # -------------------------

    chunks = [
        line.strip()
        for line in dataset.splitlines()
        if line.strip()
    ]

    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    t2 = time.perf_counter()

    # -------------------------
    # 9. Generate embeddings with Ray
    # -------------------------

    print(f"Generating embeddings with Ray (batches of {EMBED_BATCH_SIZE})...")

    vector_db = build_vector_db(chunks)

    t3 = time.perf_counter()

    print(f"Vector database created: {len(vector_db)} chunks")

    # -------------------------
    # 10. User query
    # -------------------------

    input_query = input("\nAsk me a question: ")

    # -------------------------
    # 11. Retrieve relevant chunks
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
    # 12. Build prompt
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
    # 13. Generate response
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
    # 14. Execution time (hors saisie et warm-up)
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
