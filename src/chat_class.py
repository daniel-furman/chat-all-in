import datetime
import time

import datasets

from src.inf_server import call_inf_server


# download podcast database
ds_episodes = datasets.load_dataset("dfurman/All-In-Podcast-Transcripts")
# download cache conversation databse
# ds_conversations = datasets.load_dataset("dfurman/Chat-All-In-Conversations")


class Chat:
    default_system_prompt = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
    system_format = "<|im_start|>system\n{}<|im_end|>\n"

    def __init__(
        self, system: str = None, user: str = None, assistant: str = None
    ) -> None:
        if system is not None:
            self.set_system_prompt(system)
        else:
            self.reset_system_prompt()
        self.user = user if user else "<|im_start|>user\n{}<|im_end|>\n"
        self.assistant = (
            assistant if assistant else "<|im_start|>assistant\n{}<|im_end|>\n"
        )
        self.response_prefix = self.assistant.split("{}")[0]

    def set_system_prompt(self, system_prompt):
        # self.system = self.system_format.format(system_prompt)
        return system_prompt

    def reset_system_prompt(self):
        return self.set_system_prompt(self.default_system_prompt)

    def history_as_formatted_str(self, system, history) -> str:
        system = self.system_format.format(system)
        text = system + "".join(
            [
                "\n".join(
                    [
                        self.user.format(item[0]),
                        self.assistant.format(item[1]),
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += self.user.format(history[-1][0])
        text += self.response_prefix

        # stopgap solution to too long sequences
        if len(text) > 4500:
            # delete from the middle between <|im_start|> and <|im_end|>
            # find the middle ones, then expand out
            start = text.find("<|im_start|>", 139)
            end = text.find("<|im_end|>", 139)
            while end < len(text) and len(text) > 4500:
                end = text.find("<|im_end|>", end + 1)
                text = text[:start] + text[end + 1 :]
        if len(text) > 4500:
            # the nice way didn't work, just truncate
            # deleting the beginning
            text = text[-4500:]

        return text

    def clear_history(self, history):
        return []

    def save_history(self, history):
        # Getting the current date and time
        dt = datetime.now()
        dt = str(dt).replace(" ", "-").replace(":", "-").replace(".", "-")
        return history

    def turn(self, user_input: str):
        self.user_turn(user_input)
        return self.bot_turn()

    def user_turn(self, user_input: str, history):
        history.append([user_input, ""])
        return user_input, history

    def bot_turn(self, system, history):
        conversation = self.history_as_formatted_str(system, history)
        assistant_response = call_inf_server(conversation)
        # history[-1][-1] = assistant_response
        # return "", history
        history[-1][1] = ""
        for chunk in assistant_response:
            try:
                decoded_output = chunk["choices"][0]["delta"]["content"]
                history[-1][1] += decoded_output
                yield history
            except KeyError:
                pass

    def user_turn_select_episode(self, history):
        user_input = (
            "Starter call: Display background information for the selected episode."
        )
        history.append([user_input, ""])
        return history

    def bot_turn_select_episode(self, history, episode):
        episode_num = episode.split("(")[-1].split(")")[0]
        assistant_response = f"All-In Episode {episode_num}:\n\n"
        assistant_response += f'Title: {ds_episodes[episode_num]["episode_title"][0].replace(episode_num + ": ", "")}\n'
        assistant_response += (
            f"Date aired: {ds_episodes[episode_num]['episode_date'][0]}\n"
        )
        assistant_response += "Sections:\n\n"
        for itr, section_title in enumerate(ds_episodes[episode_num]["section_title"]):
            assistant_response += f"{itr}. {section_title} ({ds_episodes[episode_num]['section_time_stamp'][itr]})\n"
        assistant_response += (
            "\nYou can now converse with the assistant about this episode!"
        )

        history[-1][1] = ""
        for character in assistant_response:
            history[-1][1] += character
            time.sleep(0.00075)
            yield history
