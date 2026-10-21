"""Step 5 — Answer visitors' questions in natural language (course 8).

The museum catalogue = one short description per artwork. A visitor types a sentence; we return
the artworks whose description is closest in meaning.

Two methods, compared on the same labelled questions:
  * classical baseline : TF-IDF (word counts weighted by rarity) + cosine similarity
  * neural             : sentence embeddings all-MiniLM-L6-v2 (384 numbers per sentence) + cosine
Metrics: top-1 and top-3 hit rate (is a correct artwork first / in the first three?) and CPU time.
Run: pip install sentence-transformers   then   python scripts/05_search.py
"""

# ---------------------------------------------------------------------------------------------
# 1. Libraries
# ---------------------------------------------------------------------------------------------
import sys                     # sys: lets us change Python's search path (used just below)
import time                    # time: precise stopwatch, to measure how long each method takes
from pathlib import Path       # Path: handles file paths; used below to locate the project folder

# Add the project folder (two levels above this file) to Python's search path,
# so that "from src.utils import ..." works.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # numpy: maths on arrays of numbers
from sklearn.feature_extraction.text import TfidfVectorizer   # TF-IDF: sentence -> word-count vector

from src.utils import save_metrics     # our helper that writes results to outputs/metrics/*.json

# ---------------------------------------------------------------------------------------------
# 2. The catalogue: (title, style, description), one entry per artwork.
#    WikiArt has no descriptions, so we write them like museum labels.
#    Each artwork's position in this list (0, 1, 2...) is its index, used in QUESTIONS below.
# ---------------------------------------------------------------------------------------------
CATALOGUE = [
    ("Hokusai — The Great Wave off Kanagawa", "Ukiyo_e",                                  # index 0
     "A huge wave with claw-like foam towers over three fishing boats; Mount Fuji appears small in the background."),
    ("Hiroshige — Sudden Shower over Shin-Ohashi Bridge", "Ukiyo_e",                      # index 1
     "People hurry across a wooden bridge under heavy slanting rain, a raft drifts on the river below."),
    ("Hokusai — Fine Wind, Clear Morning", "Ukiyo_e",                                     # index 2
     "A red mountain rises under a blue sky dotted with small white clouds."),
    ("Monet — Impression, Sunrise", "Impressionism",                                      # index 3
     "An orange sun rises over a misty harbour, small dark boats in the foreground."),
    ("Monet — Water Lilies", "Impressionism",                                             # index 4
     "Water lilies float on a pond, soft colours and reflections of light and sky."),
    ("Renoir — Bal du moulin de la Galette", "Impressionism",                             # index 5
     "A crowd dances and chats in an open-air garden in Paris, sunlight filtering through the trees."),
    ("Monet — Woman with a Parasol", "Impressionism",                                     # index 6
     "A woman with a parasol stands on a windy hill, clouds moving across the sky."),
    ("Caravaggio — The Calling of Saint Matthew", "Baroque",                              # index 7
     "In a dark room, a beam of light falls on men counting money at a table as Christ points at one of them."),
    ("Rembrandt — The Night Watch", "Baroque",                                            # index 8
     "A company of armed militiamen marches out, led by a captain in black and a lieutenant in yellow."),
    ("Vermeer — Girl with a Pearl Earring", "Baroque",                                    # index 9
     "A girl in a blue and yellow turban looks over her shoulder, a pearl earring catching the light."),
    ("Rembrandt — The Storm on the Sea of Galilee", "Baroque",                            # index 10
     "A small boat is tossed by a violent storm, sailors struggle with the sail as waves crash over the bow."),
    ("Murillo — Boys Eating Grapes and Melon", "Baroque",                                 # index 11
     "Two poor boys sit on the ground eating grapes and a slice of melon."),
    ("Picasso — Les Demoiselles d'Avignon", "Cubism",                                     # index 12
     "Five nude women with angular, mask-like faces, broken into sharp geometric planes."),
    ("Braque — Violin and Candlestick", "Cubism",                                         # index 13
     "A violin and a candlestick fragmented into overlapping grey and brown facets."),
    ("Juan Gris — Portrait of Picasso", "Cubism",                                         # index 14
     "A painter holding a palette, his face and body cut into blue and grey geometric shapes."),
    ("Warhol — Campbell's Soup Cans", "Pop_Art",                                          # index 15
     "Thirty-two canvases, each showing a can of tomato soup with a red and white label."),
    ("Lichtenstein — Drowning Girl", "Pop_Art",                                           # index 16
     "A comic-strip woman with blue hair cries among large waves, with a speech bubble."),
    ("Warhol — Marilyn Diptych", "Pop_Art",                                               # index 17
     "Fifty repeated portraits of a film star, bright colours on one side, fading black and white on the other."),
]

# ---------------------------------------------------------------------------------------------
# 3. Labelled visitor questions: (question, indices of the artworks that are a correct answer).
#    Written on purpose with OTHER words than the descriptions, as real visitors would.
#    These are our "test set" for the search: we know the right answers in advance.
# ---------------------------------------------------------------------------------------------
QUESTIONS = [
    ("I want to see a stormy sea", [10, 0, 16]),            # Rembrandt storm, Great Wave, Drowning Girl
    ("Portraits of children", [11]),                        # Murillo's boys
    ("A painting where it is raining", [1]),                # Hiroshige's shower
    ("The sun coming up over the water", [3]),              # Monet's sunrise
    ("People dancing outdoors", [5]),                       # Renoir's ball
    ("Flowers on a lake", [4]),                             # Monet's water lilies
    ("Soldiers carrying guns", [8]),                        # Rembrandt's Night Watch
    ("Food you could buy at the supermarket", [15]),        # Warhol's soup cans
    ("A crying woman from a comic book", [16]),             # Lichtenstein
    ("A musical instrument split into shapes", [13]),       # Braque's violin
    ("A young woman wearing jewellery", [9]),               # Vermeer
    ("A famous actress shown again and again", [17]),       # Warhol's Marilyn
    ("A religious scene with dramatic lighting", [7]),      # Caravaggio
    ("A lady with an umbrella in the wind", [6]),           # Monet's parasol
]


# ---------------------------------------------------------------------------------------------
# 4. Helper: cosine similarity between every question and every description
# ---------------------------------------------------------------------------------------------
def cosine_matrix(A, B):
    """cos(u, v) = u.v / (|u| |v|) for every row of A against every row of B."""
    # Divide each row by its length, so every vector has length 1 (clip avoids dividing by 0)
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
    # With length-1 vectors, the cosine is just the dot product; A @ B.T computes all of them:
    # result[i, k] = cosine between question i and description k
    return A @ B.T


# ---------------------------------------------------------------------------------------------
# 5. Helper: score a similarity matrix against the known right answers
# ---------------------------------------------------------------------------------------------
def evaluate(sim):
    """sim: (n_questions, n_artworks). Returns top-1 / top-3 hit rates and the ranked answers."""
    # For each question, artwork indices sorted from most to least similar
    # (-sim: sort in decreasing order; "stable": ties keep their catalogue order)
    ranking = np.argsort(-sim, axis=1, kind="stable")
    # A similarity of 0 means "no word in common": that is no answer, not a lucky first place
    found = lambda i, k: sim[i, k] > 0
    # top-1: share of questions whose FIRST proposed artwork is a right answer
    top1 = np.mean([ranking[i, 0] in good and found(i, ranking[i, 0])
                    for i, (_, good) in enumerate(QUESTIONS)])
    # top-3: share of questions with at least one right answer among the THREE first proposals
    top3 = np.mean([any(k in good and found(i, k) for k in ranking[i, :3])
                    for i, (_, good) in enumerate(QUESTIONS)])
    return float(top1), float(top3), ranking


# ---------------------------------------------------------------------------------------------
# 6. Helper: print, for each question, the artwork proposed first
# ---------------------------------------------------------------------------------------------
def show(name, sim, ranking):
    print(f"\n=== {name}")                                      # title of the method
    for i, (q, good) in enumerate(QUESTIONS):                   # for each question...
        k = ranking[i, 0]                                       # ...index of the best artwork
        mark = "OK " if (k in good and sim[i, k] > 0) else "-- "   # right or wrong?
        print(f"{mark}{q:42s} -> {CATALOGUE[k][0]:45s} (cos {sim[i, k]:.2f})")


# ---------------------------------------------------------------------------------------------
# 7. Main program
# ---------------------------------------------------------------------------------------------
def main():
    descriptions = [d for _, _, d in CATALOGUE]     # keep only the description texts (18 sentences)
    questions = [q for q, _ in QUESTIONS]           # keep only the question texts (14 sentences)
    results = {"n_artworks": len(CATALOGUE),        # results notebook: catalogue size
               "n_questions": len(QUESTIONS)}       # and number of test questions

    # ---- 7.1 Classical method: TF-IDF
    t0 = time.perf_counter()                        # start the stopwatch
    tfidf = TfidfVectorizer(stop_words="english")   # ignore very common words (the, a, of...)
    D = tfidf.fit_transform(descriptions).toarray() # learn the vocabulary on the catalogue and turn
                                                    # each description into one number per word
    Q = tfidf.transform(questions).toarray()        # turn each question into the same word columns
                                                    # (words absent from the catalogue are dropped)
    sim_tfidf = cosine_matrix(Q, D)                 # cosine between every question and description
    t_tfidf = time.perf_counter() - t0              # stop the stopwatch
    top1, top3, rank = evaluate(sim_tfidf)          # compare with the known right answers
    show("TF-IDF", sim_tfidf, rank)                 # print each question with its first answer
    results["tfidf"] = {"top1": top1, "top3": top3,                     # hit rates
                        "vocabulary_size": len(tfidf.vocabulary_),      # number of word columns
                        "total_time_s": t_tfidf,                        # total time in seconds
                        # questions whose vector is all zeros = no word in common with the catalogue
                        "questions_with_no_shared_word": int((Q.sum(axis=1) == 0).sum())}

    # ---- 7.2 Neural method: sentence embeddings
    try:                                            # try to load the sentence-transformers library
        from sentence_transformers import SentenceTransformer
    except ImportError:                             # if it is not installed: say so, save, stop
        print("\nsentence-transformers missing: pip install sentence-transformers")
        save_metrics("search", results)
        return
    t0 = time.perf_counter()                        # stopwatch: loading the model
    # Pretrained network (6 layers, 22.7 M parameters) that turns a sentence into 384 numbers;
    # downloaded from the Internet the first time
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    t_load = time.perf_counter() - t0               # time to load the model
    t0 = time.perf_counter()                        # stopwatch: encoding the catalogue
    E_desc = st.encode(descriptions)                # (18, 384): done once, stored on the kiosk
    t_index = time.perf_counter() - t0              # time to encode the whole catalogue
    t0 = time.perf_counter()                        # stopwatch: encoding the questions
    E_q = st.encode(questions)                      # (14, 384): done for every visitor question
    t_query = (time.perf_counter() - t0) / len(questions)   # average time for ONE question
    sim_emb = cosine_matrix(E_q, E_desc)            # cosine between every question and description
    top1, top3, rank = evaluate(sim_emb)            # compare with the known right answers
    show("Sentence embeddings (all-MiniLM-L6-v2)", sim_emb, rank)
    results["embeddings"] = {"top1": top1, "top3": top3,          # hit rates
                             "dim": int(E_q.shape[1]),            # 384 numbers per sentence
                             "load_time_s": t_load,               # model loading time
                             "index_time_s": t_index,             # catalogue encoding time
                             "ms_per_question": t_query * 1000,   # answer time per question, in ms
                             "n_params": sum(p.numel() for p in st.parameters())}  # model size

    # ---- 7.3 Summary on screen and save
    print(f"\nTF-IDF     top-1 {results['tfidf']['top1']:.0%}  top-3 {results['tfidf']['top3']:.0%}")
    print(f"Embeddings top-1 {results['embeddings']['top1']:.0%}  top-3 {results['embeddings']['top3']:.0%}"
          f"  ({results['embeddings']['ms_per_question']:.1f} ms / question)")
    save_metrics("search", results)                 # write outputs/metrics/search.json


# Run main() only when the file is launched directly: python scripts/05_search.py
if __name__ == "__main__":
    main()
