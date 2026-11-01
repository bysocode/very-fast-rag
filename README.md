# very-fast-rag

RAG (Retrieval-Augmented Generation) minimaliste en Python : embeddings via **Ollama**, recherche par similarité cosinus, puis génération de la réponse par un LLM local.

Le projet contient deux implémentations :

- `main.py` : version **séquentielle** ;
- `main_ray.py` : version **parallèle avec Ray** pour les embeddings.

Les deux scripts utilisent les mêmes modèles, le même dataset, le même découpage et le même prompt, afin de comparer les temps d'exécution.

## Sommaire

- [Prérequis](#prérequis)
- [Modèles Ollama](#modèles-ollama)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fonctionnement](#fonctionnement)
- [Résultats du benchmark](#résultats-du-benchmark)
- [Problèmes rencontrés et solutions](#problèmes-rencontrés-et-solutions)
- [Limites](#limites)
- [Pistes d'amélioration](#pistes-damélioration)

## Prérequis

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installé et démarré

## Modèles Ollama

```bash
ollama pull embeddinggemma:300m
ollama pull gemma3:270m
```

| Rôle | Modèle | Détail |
|---|---|---|
| Embeddings | `embeddinggemma:300m` | 768 dimensions |
| Génération | `gemma3:270m` | modèle de chat |

## Installation

```bash
uv sync
```

## Utilisation

```bash
uv run main.py
uv run main_ray.py
```

Le script demande une question, puis :

1. charge le dataset ;
2. découpe le texte en chunks (une ligne non vide = un chunk) ;
3. calcule les embeddings ;
4. récupère les `TOP_N = 3` chunks les plus proches ;
5. construit le prompt et génère la réponse.

### Test rapide

Pour éviter de traiter tout le dataset à chaque essai :

```bash
MAX_CHUNKS=512 uv run main.py
MAX_CHUNKS=512 uv run main_ray.py
```

`MAX_CHUNKS=0` (valeur par défaut) traite tout le dataset.

### Datasets

- `HarryPotter.txt` : dataset par défaut, **7 646 chunks** ;
- `bible.txt` : **31 105 chunks**, utile pour tester des questions bibliques (par exemple « Who is Cain? »).

Modifier `DATASET_PATH` dans les deux fichiers pour changer de dataset.

## Configuration

Les constantes sont identiques dans `main.py` et `main_ray.py` :

```python
DATASET_PATH = "HarryPotter.txt"
EMBEDDING_MODEL = "embeddinggemma:300m"
LANGUAGE_MODEL = "gemma3:270m"
TOP_N = 3
EMBED_BATCH_SIZE = 32
```

- `EMBED_BATCH_SIZE` : nombre de chunks envoyés dans un même appel `ollama.embed` ;
- `MAX_CHUNKS` : variable d'environnement optionnelle pour limiter le nombre de chunks.

## Fonctionnement

```text
dataset
   │
   ▼
découpage en chunks
   │
   ▼
embeddings (Ollama)
   │
   ▼
similarité cosinus (NumPy)
   │
   ▼
top 3 chunks
   │
   ▼
prompt + génération (Ollama)
```

La version Ray distribue les lots d'embeddings sur les workers :

```text
chunks → lots de 32 → tâches Ray → embeddings → base vectorielle
```

## Résultats du benchmark

### Environnement de test

| Élément | Valeur |
|---|---|
| CPU | 12 cœurs |
| GPU | NVIDIA GeForce RTX 4060 (8 Go) |
| Ollama | 0.40.13 |
| Ray | 2.59.0 |
| Python | 3.14 |
| Dataset | `HarryPotter.txt`, 7 646 chunks |

Les temps ci-dessous sont mesurés hors temps de saisie et hors warm-up. Les résultats dépendent du matériel et de l'état d'Ollama.

### Avant optimisation

Version initiale : un appel Ollama par chunk, retrieval en boucle Python, retrieval Ray avec une tâche par chunk, pas de warm-up.

| Phase | Séquentiel | Ray |
|---|---:|---:|
| Ray init | — | 2,15 s |
| Embeddings | 41,66 s | 51,65 s |
| Retrieval | 0,56 s | 4,07 s |
| LLM | 4,22 s | 0,03 s |
| **Total** | **46,44 s** | **57,90 s** |

Problèmes visibles :

- Ray est **plus lent** que le séquentiel ;
- le retrieval Ray coûte 4,07 s pour un simple calcul de similarité ;
- le temps LLM n'est pas comparable : le premier run charge le modèle, le second le trouve déjà chaud.

### Après optimisation

Optimisations appliquées : embeddings par lots de 32, retrieval vectorisé NumPy, warm-up des modèles.

| Phase | Séquentiel | Ray |
|---|---:|---:|
| Ray init | — | 2,22 s |
| Embeddings | 36,89 s | 35,46 s |
| Retrieval | 0,12 s | 0,41 s |
| LLM | 0,07 s | 0,03 s |
| **Total** | **37,08 s** | **38,12 s** |

### Test rapide (512 chunks)

| Phase | Séquentiel | Ray |
|---|---:|---:|
| Ray init | — | 2,24 s |
| Embeddings | 2,47 s | 2,69 s |
| Retrieval | 0,01 s | 0,02 s |
| LLM | 0,06 s | 0,02 s |
| **Total** | **2,55 s** | **4,97 s** |

### Interprétation

- Les **embeddings par lots** réduisent l'overhead HTTP : 41,66 → 36,89 s en séquentiel.
- Le **retrieval vectorisé** fait passer le séquentiel de 0,56 à 0,12 s et Ray de 4,07 à 0,41 s.
- **Ray n'apporte pas de gain sur les embeddings** avec un seul serveur Ollama : Ollama sérialise les requêtes d'embedding.
- Sur un petit dataset, le coût d'initialisation de Ray (~2,2 s) domine largement.
- Le **warm-up** rend les temps LLM comparables entre les deux exécutions.

## Problèmes rencontrés et solutions

### 1. Warning `VIRTUAL_ENV` avec Ray + `uv run`

Message observé :

```text
(raylet) warning: `VIRTUAL_ENV=/home/.../.venv` does not match the project
environment path `.venv` and will be ignored; use `--active` to target the
active environment instead [repeated 4x across cluster]
```

**Cause** : Ray 2.59 détecte que le driver est lancé avec `uv run` et propage l'environnement uv aux workers. Il relance `uv run` dans le dossier empaqueté par Ray, où le venv est `.venv` (relatif), alors que `VIRTUAL_ENV` pointe vers un chemin absolu. uv considère qu'il y a un mismatch, ignore `VIRTUAL_ENV`, affiche le warning et recrée un venv pour chaque worker.

**Solution** dans `main_ray.py`, avant `import ray` :

```python
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
```

Les workers utilisent alors directement le Python du venv courant, sans recréer d'environnement.

> Le warning `Token authentication is enabled...` est indépendant. Pour le masquer : `export RAY_AUTH_MODE=disabled`.

### 2. Ollama sérialise les embeddings (`-np 1`)

**Constat** : le processus `llama-server` est lancé avec `-np 1` pour les modèles d'embedding. Un test avec un second serveur Ollama et `OLLAMA_NUM_PARALLEL=4` donne :

- embedding : toujours `-np 1` ;
- chat : `-np 4`.

**Conséquence** : Ray ne peut pas paralléliser les embeddings via un seul serveur Ollama. Les workers Ray attendent le serveur, l'un après l'autre.

**Pistes** :

- plusieurs serveurs Ollama (un par GPU/CPU) et répartition des lots ;
- backend d'embedding local dans les workers Ray (`sentence-transformers`, `llama-cpp-python`) ;
- conserver le séquentiel par lots pour les embeddings.

### 3. Retrieval via Ray plus lent que NumPy

**Cause** : une tâche Ray par chunk (7 646 tâches) pour un simple produit scalaire, avec sérialisation et scheduling.

**Solution** : calcul vectorisé avec NumPy dans les deux scripts :

```python
similarities = matrix @ query_embedding / (norms * query_norm)
```

### 4. Temps LLM faussé par le chargement du modèle

Le premier appel à `ollama.chat` charge le modèle en mémoire (plusieurs secondes). Le second run le trouve déjà chaud, d'où un temps LLM artificiellement très bas.

**Solution** : fonction `warmup()` appelée avant les timings, qui charge les modèles d'embedding et de chat.

### 5. « Who is cain ? » sur Harry Potter

Cain n'apparaît pas dans Harry Potter. La réponse « I don't know » est donc **correcte**. Les chunks récupérés sont des fragments de dialogue sans rapport avec la question.

Pour tester Cain, utiliser `bible.txt` à la place de `HarryPotter.txt`.

### 6. Chunking ligne par ligne

Le découpage actuel (une ligne non vide = un chunk) produit beaucoup de fragments très courts (« him? », « he is? », « Who? »), ce qui dégrade la qualité du retrieval.

**Piste** : regrouper les lignes en paragraphes de ~500 caractères avec un chevauchement.

## Limites

- Pas d'index vectoriel : la recherche est exhaustive et en mémoire.
- Chunking naïf (ligne par ligne).
- Pas de cache d'embeddings sur disque.
- Pas de gestion de la mémoire pour de très gros datasets.
- Les benchmarks dépendent du matériel et de l'état d'Ollama.
- `bible.txt` (31 105 chunks) est beaucoup plus long à indexer que `HarryPotter.txt`.

## Pistes d'amélioration

- Chunking par paragraphes ou phrases avec overlap.
- Index vectoriel (FAISS, Qdrant, etc.).
- Lots d'embeddings plus grands.
- Backend d'embedding local dans les workers Ray.
- Cache des embeddings sur disque.
- Évaluation de la qualité du retrieval (`recall@k`).
