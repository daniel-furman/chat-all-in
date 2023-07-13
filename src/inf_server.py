import logging
import time
from src.llm_boilers import llm_boiler


def call_inf_server(prompt, openai_key, episode_number):
    model_id = "gpt-3.5-turbo-16k"
    model = llm_boiler(model_id, openai_key)

    logging.warning(f'Inference via "{model_id}"" for prompt "{prompt}"')

    try:
        # run text generation
        response = model.run(
            prompt,
            temperature=1.0,
            n_answers=5,
            episode_number=episode_number,
        )

        logging.warning(f"Result of text generation: {response}")
        return response

    except Exception as e:
        # assume it is our error
        # just wait and try one more time
        print(e)
        time.sleep(2)
        response = model.run(
            prompt,
            temperature=1.0,
            n_answers=5,
            episode_number=episode_number,
        )
        logging.warning(f"Result of text generation: {response}")
        return response
