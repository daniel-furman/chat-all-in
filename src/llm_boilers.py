# custom text generation llm classes

import warnings
import logging
import numpy as np
import datasets
import openai

from src.semantic_search import basic_semantic_search

# supress warnings
warnings.filterwarnings("ignore")


class llm_boiler:
    def __init__(self, model_id, openai_key):
        self.model_id = model_id
        self.openai_key = openai_key
        for f_idx, run_function in enumerate(MODEL_FUNCTIONS):
            if run_function.__name__.lower() in self.model_id:
                print(
                    f"Load function recognized for {self.model_id}: {LOAD_MODEL_FUNCTIONS[f_idx].__name__}"
                )
                self.load_fn = LOAD_MODEL_FUNCTIONS[f_idx]
        for run_function in MODEL_FUNCTIONS:
            if run_function.__name__.lower() in self.model_id:
                print(
                    f"Run function recognized for {self.model_id}: {run_function.__name__.lower()}"
                )
                self.run_fn = run_function
        self.model = self.load_fn(self.model_id, self.openai_key)
        self.name = self.run_fn.__name__.lower()

    def run(
        self,
        prompt,
        temperature,
        n_answers,
        episode_number,
    ):
        return self.run_fn(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            n_answers=n_answers,
            episode_number=episode_number,
        )


LOAD_MODEL_FUNCTIONS = []
MODEL_FUNCTIONS = []


# gpt models
def gpt_loader(model_id: str, openai_key: str):
    # Load your API key from an environment variable or secret management service
    openai.api_key = openai_key  # os.getenv("OPENAI_API_KEY")
    logging.warning(f"model id: {model_id}")

    return model_id


LOAD_MODEL_FUNCTIONS.append(gpt_loader)


def gpt(
    model: str,
    prompt: str,
    temperature: int,
    n_answers: int,
    episode_number: str,
) -> str:
    """
    Initialize the pipeline
    Uses Hugging Face GenerationConfig defaults
        https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig
    Args:
        model (str): openai model key
        tokenizer (str): openai model key
        prompt (str): Prompt for text generation
        max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0
    """
    ds_episodes = datasets.load_dataset(
        "dfurman/All-In-Podcast-Transcripts", split=episode_number
    )
    df_episodes = ds_episodes.to_pandas()

    conversation = prompt.split("<|im_start|>")

    messages = []
    search_query = ""
    user_itr = 0
    for turn in conversation:
        first_word = turn.split("\n")[0]

        if first_word == "system":
            messages.append(
                {
                    "role": "system",
                    "content": turn.replace("system\n", "").replace("<|im_end|>\n", ""),
                }
            )
        elif first_word == "user":
            messages.append(
                {
                    "role": "user",
                    "content": turn.replace("user\n", "").replace("<|im_end|>\n", ""),
                }
            )

            if user_itr != 0:
                search_query += (
                    turn.replace("user\n", "").replace("<|im_end|>\n", "") + " "
                )
            user_itr += 1
        elif first_word == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.replace("assistant\n", "").replace(
                        "<|im_end|>\n", ""
                    ),
                }
            )
    # drop empty last element from above
    messages = messages[0 : len(messages) - 1]

    # retreive context
    logging.warning(f"SEMANTIC SEARCH QUERY: {search_query}")
    # first try hard coded section number mention
    included_context = None
    for i in range(len(df_episodes)):
        if f"section {i+1}" in search_query.lower():
            included_context_dialogue = df_episodes.iloc[i]["section_dialogue"]
            section_hit = df_episodes.iloc[i]["section_title"]
            included_context = f"{section_hit}: {included_context_dialogue}"
    # if no hits above, run semantic search against sentence embeddings
    if included_context is None:
        corpus_texts_metadata_ordered = basic_semantic_search(
            search_query,
            n_answers,
            episode_number,
        )
        top_hit = corpus_texts_metadata_ordered.iloc[0]
        logging.warning(f"SEMANTIC SEARCH TOP SENTENCE GRABBED: {top_hit.sentences}")
        section_hit = top_hit.section_title
        included_context_dialogue = df_episodes[
            df_episodes["section_title"] == section_hit
        ]["section_dialogue"].iloc[0]
        included_context = f"{section_hit}: {included_context_dialogue}"

    # format in-context prompt
    messages[-1]["content"] = (
        messages[-1]["content"]
        + "\n\n"
        + "Here's some of the episode's transcript, which may contain relavent information for your response."
        + "\n\n"
        + f'"{included_context}"'
        + "\n\n"
        + "When possible, do not repeat information that has already been said in the above assistant responses."
        + "\n\n"
        + "Where appropriate, think this through in a step by step manner to make sure we have the right answer."
    )
    logging.warning(f"INPUT TO OPENAI CALL AFTER CONTEXT: {messages}\n")

    # init streaming chat completion
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    return chat_completion


MODEL_FUNCTIONS.append(gpt)
