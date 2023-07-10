# custom text generation llm classes

import warnings
import logging
import os

import openai

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
    ):
        return self.run_fn(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
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
    conversation = prompt.split("<|im_start|>")

    messages = []
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
        elif first_word == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.replace("assistant\n", "").replace(
                        "<|im_end|>\n", ""
                    ),
                }
            )

    logging.warning(f"Input to openai api call: {messages}")

    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    return chat_completion


MODEL_FUNCTIONS.append(gpt)
