import logging
import time
from src.llm_boilers import llm_boiler


def call_inf_server(prompt, openai_key):
    model_id = "gpt-3.5-turbo"  # "gpt-3.5-turbo-16k",
    model = llm_boiler(model_id, openai_key)

    logging.warning(f'Inf via "{model_id}"" for prompt "{prompt}"')

    try:
        # add context to prompt
        # start by just doing summary
        # corpus_texts = basic_semantic_search(prompt, n_answers=1, episode_number="E134")
        # logging.warning("Corpus text returned by semantic search: ", corpus_texts)

        # run text generation
        response = model.run(prompt, temperature=1.0)
        logging.warning(f"Result of text generation: {response}")
        return response

    except Exception as e:
        # assume it is our error
        # just wait and try one more time
        print(e)
        time.sleep(2)
        response = model.run(prompt, temperature=1.0)
        logging.warning(f"Result of text generation: {response}")
        return response
