import os
from argparse import ArgumentParser
from torch.nn.functional import cosine_similarity
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# python basic_semantic_search.py --query "tiger global" --n_answers 1 --episode_number E134


def main(
    query: str,
    n_answers: int,
    episode_number: str,
):
    # change to project root dir
    os.chdir("../..")
    model_name = "sentence-transformers/all-mpnet-base-v2"

    corpus_texts = corpus_texts = pd.read_parquet(
        f"data/all-in-transcripts/cleaned/{episode_number}_sections_full_cleaned.parquet"
    )
    model = SentenceTransformer(model_name)

    corpus_emb = torch.randn(len(corpus_texts.section_summary), 768)
    for itr, text in enumerate(corpus_texts["section_summary"]):
        corpus_emb[itr, :] = model.encode(text, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)

    # Getting hits
    hits = cosine_similarity(query_emb[None, :], corpus_emb, dim=1, eps=1e-8)

    corpus_texts["similarity"] = hits.tolist()

    # filter to just top n answers
    corpus_texts = corpus_texts.sort_values(by="similarity", ascending=False).head(
        n_answers
    )

    print(corpus_texts)
    return corpus_texts


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
        help="Episode number, example: E132",
    ),
    args = parser.parse_args()
    main(args.query, args.n_answers, args.episode_number)
