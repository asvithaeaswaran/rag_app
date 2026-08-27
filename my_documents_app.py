def search_similar_chunks(
    query,
    top_k=TOP_K,
    api_key=None,
    min_score=0.25
):
    """
    Retrieve relevant document chunks using cosine similarity.

    A lower similarity threshold is used so that valid information
    is not rejected too early.
    """

    global chunk_vectors
    global chunks_registry

    # -------------------------------------------------------------------------
    # CHECK DOCUMENTS / VECTORS
    # -------------------------------------------------------------------------

    if (
        not chunks_registry
        or chunk_vectors is None
        or len(chunk_vectors) == 0
    ):
        print("No document chunks or vectors available.")
        return []

    # -------------------------------------------------------------------------
    # GET GEMINI API KEY
    # -------------------------------------------------------------------------

    gemini_key = sanitize_key(
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    query_vector = None

    # -------------------------------------------------------------------------
    # GEMINI QUERY EMBEDDING
    # -------------------------------------------------------------------------

    if (
        gemini_key
        and len(chunk_vectors.shape) == 2
        and chunk_vectors.shape[1] == 768
    ):

        try:

            raw_vector = get_gemini_embedding(
                query,
                gemini_key
            )

            if raw_vector:

                query_vector = np.array(
                    raw_vector,
                    dtype=np.float32
                )

                print(
                    "Using Gemini embedding for query."
                )

        except Exception as e:

            print(
                "Gemini query embedding error:",
                e
            )

            query_vector = None

    # -------------------------------------------------------------------------
    # TF-IDF FALLBACK
    # -------------------------------------------------------------------------

    if query_vector is None:

        print(
            "Using TF-IDF fallback for query."
        )

        all_texts = [
            chunk["text"]
            for chunk in chunks_registry
        ]

        vocabulary, idf = build_tfidf_vocabulary(
            all_texts
        )

        query_vector = compute_tfidf_vector(
            query,
            vocabulary,
            idf
        )

    # -------------------------------------------------------------------------
    # VALIDATE VECTOR DIMENSION
    # -------------------------------------------------------------------------

    if query_vector is None:

        print(
            "Could not create query vector."
        )

        return []

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    if query_vector.ndim != 1:

        query_vector = query_vector.flatten()

    if (
        len(chunk_vectors.shape) != 2
        or chunk_vectors.shape[1] != len(query_vector)
    ):

        print(
            "Vector dimension mismatch."
        )

        print(
            "Document vector dimension:",
            chunk_vectors.shape
        )

        print(
            "Query vector dimension:",
            len(query_vector)
        )

        return []

    # -------------------------------------------------------------------------
    # COSINE SIMILARITY
    # -------------------------------------------------------------------------

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:

        print(
            "Query vector is empty."
        )

        return []

    chunk_norms = np.linalg.norm(
        chunk_vectors,
        axis=1
    )

    denominator = (
        chunk_norms
        *
        query_norm
    )

    denominator = np.maximum(
        denominator,
        1e-8
    )

    scores = (
        np.dot(
            chunk_vectors,
            query_vector
        )
        /
        denominator
    )

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # -------------------------------------------------------------------------
    # RANK RESULTS
    # -------------------------------------------------------------------------

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    print(
        "=================================================="
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Total chunks available: {len(chunks_registry)}"
    )

    print(
        f"Similarity threshold: {min_score}"
    )

    # -------------------------------------------------------------------------
    # COLLECT RELEVANT CHUNKS
    # -------------------------------------------------------------------------

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        # Ignore very weak matches
        if score < min_score:
            continue

        if index >= len(
            chunks_registry
        ):
            continue

        chunk = chunks_registry[index]

        results.append(
            {
                "rank": len(results) + 1,

                "filename": chunk.get(
                    "filename",
                    "Unknown"
                ),

                "page": chunk.get(
                    "page",
                    0
                ),

                "chunk_index": chunk.get(
                    "chunk_index",
                    index
                ),

                "text": chunk.get(
                    "text",
                    ""
                ),

                "score": round(
                    score,
                    4
                ),

                "snippet": (
                    chunk.get(
                        "text",
                        ""
                    )[:300]
                    +
                    (
                        "..."
                        if len(
                            chunk.get(
                                "text",
                                ""
                            )
                        ) > 300
                        else ""
                    )
                )
            }
        )

        if len(results) >= top_k:
            break

    # -------------------------------------------------------------------------
    # DEBUG INFORMATION
    # -------------------------------------------------------------------------

    print(
        f"Retrieved chunks: {len(results)}"
    )

    if len(ranked_indices) > 0:

        print(
            "Top similarity scores:"
        )

        for i in ranked_indices[:5]:

            print(
                f"  Score={float(scores[i]):.4f} "
                f"Index={i}"
            )

    for result in results:

        print(
            f"  Selected: "
            f"Score={result['score']} "
            f"Page={result['page']} "
            f"File={result['filename']}"
        )

    print(
        "=================================================="
    )

    return results