import os
from argparse import ArgumentParser
import logging

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from torch.nn.functional import cosine_similarity
from torch import from_numpy


# python basic_semantic_search.py --query "tiger global" --n_answers 1 --episode_number E134


def basic_semantic_search(
    query: str,
    n_answers: int,
    episode_number: str,
):
    embedding_model_name = "cached-all-mpnet-base-v2"
    model = SentenceTransformer(embedding_model_name)

    corpus_texts_metadata = pd.read_parquet(
        f"./embeddings/{episode_number}_sentence_embeddings_metadata.parquet"
    )

    corpus_emb = np.load(f"./embeddings/{episode_number}_sentence_embeddings.npy")
    corpus_emb = from_numpy(corpus_emb)
    query_emb = model.encode(query, convert_to_tensor=True)

    # Getting hits
    hits = cosine_similarity(query_emb[None, :], corpus_emb, dim=1, eps=1e-8)

    corpus_texts_metadata["similarity"] = hits.tolist()

    # Filter to just top n answers
    corpus_texts_metadata_ordered = corpus_texts_metadata.sort_values(
        by="similarity", ascending=False
    ).head(n_answers)

    logging.warning(
        f"SEM SEARCH TOP N SENTENCES: {corpus_texts_metadata_ordered.sentences}"
    )
    return corpus_texts_metadata_ordered


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Search query",
    ),
    parser.add_argument("--n_answers", type=int, help="N hits returned"),
    parser.add_argument(
        "--episode_number",
        type=str,
        help="Episode number, example: E134",
    ),
    args = parser.parse_args()
    basic_semantic_search(args.query, args.n_answers, args.episode_number)
