import os
import pathlib
import time

import numpy as np
import ollama

# Modèles identiques dans main.py et main_ray.py pour une comparaison juste
DATASET_PATH = "HarryPotter.txt"
EMBEDDING_MODEL = "embeddinggemma:300m"
LANGUAGE_MODEL = "gemma3:270m"
TOP_N = 3

# Un appel Ollama pour plusieurs chunks : moins d'overhead HTTP
EMBED_BATCH_SIZE = 32

# Permet un test rapide : MAX_CHUNKS=512 uv run main.py
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "0"))  # 0 = tout le dataset

VECTOR_DB = []


# -------------------------
# 1. Initialisation
# -------------------------

def create_dataset(filepath):
    path = pathlib.Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def add_chunks_to_database(chunks):
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]

        embeddings = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch
        )["embeddings"]

        VECTOR_DB.extend(zip(batch, embeddings))


def retrieve(query, top_n=TOP_N):
    query_embedding = np.asarray(
        ollama.embed(
            model=EMBEDDING_MODEL,
            input=query
        )["embeddings"][0],
        dtype=np.float32,
    )

    matrix = np.asarray(
        [embedding for _, embedding in VECTOR_DB],
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
        (VECTOR_DB[i][0], float(similarities[i]))
        for i in indices
    ]


def main():
    # -------------------------
    # 2. Warm-up (hors timings)
    # -------------------------

    warmup()

    # -------------------------
    # 3. Load dataset
    # -------------------------

    t0 = time.perf_counter()
    dataset = create_dataset(DATASET_PATH)
    t1 = time.perf_counter()

    # -------------------------
    # 4. Split dataset into chunks
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
    # 5. Generate embeddings
    # -------------------------

    print(f"Generating embeddings sequentially (batches of {EMBED_BATCH_SIZE})...")

    add_chunks_to_database(chunks)

    t3 = time.perf_counter()

    print(f"Vector database created: {len(VECTOR_DB)} chunks")

    # -------------------------
    # 6. User query
    # -------------------------

    input_query = input("\nAsk me a question: ")

    # -------------------------
    # 7. Retrieve relevant chunks
    # -------------------------

    t_retrieval_start = time.perf_counter()
    retrieved_knowledge = retrieve(input_query)
    t4 = time.perf_counter()

    print("\nRetrieved knowledge:")

    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")

    # -------------------------
    # 8. Build prompt
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
    # 9. Generate response
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
        print(chunk["message"]["content"], end="", flush=True)

    t5 = time.perf_counter()

    # -------------------------
    # 10. Execution time (hors saisie et warm-up)
    # -------------------------

    retrieval_time = t4 - t_retrieval_start
    total_time = (
        (t1 - t0)
        + (t2 - t1)
        + (t3 - t2)
        + retrieval_time
        + (t5 - t4)
    )

    print("\n\n========== EXECUTION (SÉQUENTIEL) ==========")
    print(f"Dataset loading:       {t1 - t0:.2f}s")
    print(f"Chunking:              {t2 - t1:.2f}s")
    print(f"Embedding generation:  {t3 - t2:.2f}s")
    print(f"Retrieval:             {retrieval_time:.2f}s")
    print(f"LLM generation:        {t5 - t4:.2f}s")
    print("---------------------------------------------")
    print(f"Total (hors saisie):   {total_time:.2f}s")


if __name__ == "__main__":
    main()
