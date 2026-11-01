import pathlib
import time

import numpy as np
import ollama

# Modèles identiques dans main.py et main_ray.py pour une comparaison juste
DATASET_PATH = "HarryPotter.txt"
EMBEDDING_MODEL = "embeddinggemma:300m"
LANGUAGE_MODEL = "gemma3:270m"
TOP_N = 3

VECTOR_DB = []


# -------------------------
# 1. Initialisation
# -------------------------

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def create_dataset(filepath):
    path = pathlib.Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def add_chunk_to_database(chunk):
    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )["embeddings"][0]

    VECTOR_DB.append((chunk, embedding))


def retrieve(query, top_n=TOP_N):
    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )["embeddings"][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


def main():
    # -------------------------
    # 2. Load dataset
    # -------------------------

    t0 = time.perf_counter()
    dataset = create_dataset(DATASET_PATH)
    t1 = time.perf_counter()

    # -------------------------
    # 3. Split dataset into chunks
    # -------------------------

    chunks = [
        line.strip()
        for line in dataset.splitlines()
        if line.strip()
    ]
    t2 = time.perf_counter()

    # -------------------------
    # 4. Generate embeddings
    # -------------------------

    print("Generating embeddings sequentially...")

    for chunk in chunks:
        add_chunk_to_database(chunk)

    t3 = time.perf_counter()

    print(f"Vector database created: {len(VECTOR_DB)} chunks")

    # -------------------------
    # 5. User query
    # -------------------------

    input_query = input("\nAsk me a question: ")

    # -------------------------
    # 6. Retrieve relevant chunks
    # -------------------------

    t_retrieval_start = time.perf_counter()
    retrieved_knowledge = retrieve(input_query)
    t4 = time.perf_counter()

    print("\nRetrieved knowledge:")

    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")

    # -------------------------
    # 7. Build prompt
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
    # 8. Generate response
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
    # 9. Execution time (hors temps de saisie)
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
