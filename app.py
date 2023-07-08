import os
import logging
import gradio as gr

from src.chat_class import Chat


# Logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.warning("START: App running...")


EPISODES = [
    "Jun 23, 2023: Ukraine counteroffensive, China tensions, COVID Patient Zero, RFK Jr reaction & more (E134)",
]

if os.environ.get("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown(
        """<h1><center>Chat with the All In Podcast</center></h1>

        This is a demo of a chatbot that knows up-to-date M&A news from the [All In](https://www.youtube.com/channel/UCESLZhusAkFfsNsApnjF_Cg) podcast. Start by selecting an episode of interest below - and you're off 🚀.

"""
    )
    # to do: change to openaikey input for public release
    # openai_key = gr.Textbox(
    # label="OpenAI Key",
    # value="",
    # type="password",
    # placeholder="sk..",
    # info = "You have to provide your own openai API key.",
    # )
    conversation = Chat()
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            select_episode = gr.Dropdown(
                EPISODES,
                label="Select Episode",
                info="Will add more episodes later!",
            )
    with gr.Row():
        with gr.Column(scale=2):
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    system = gr.Textbox(
                        label="System Prompt",
                        value=Chat.default_system_prompt,
                        show_label=False,
                    ).style(container=True)
                with gr.Column():
                    with gr.Row():
                        change = gr.Button("Change System Prompt")
                        reset = gr.Button("Reset System Prompt")
            with gr.Row():
                save_history = gr.Button("Cache Ideal Conversation History")

    with gr.Row():
        gr.Markdown(
            "Disclaimer: Chat-All-In can produce factually incorrect output "
            "and should not be solely relied on to produce factually accurate information. The LLMs were trained on "
            "various internet datasets; while great efforts were taken to clean the pretraining data, it is "
            "possible that these models could generate lewd, biased, or otherwise offensive outputs. Additionally, while "
            "context retrieval is used to mitigate such errors, this method can itself lead to problems for edge cases.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[chatbot],
        queue=True,
    )
    # still need to edit below -> add special prompt catch in generation for displaying sections
    grab_sections_select_event = select_episode.select(
        fn=conversation.user_turn_select_episode,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn_select_episode,
        inputs=[chatbot, select_episode],
        outputs=[chatbot],
        queue=True,
    )
    save_history.click(
        fn=conversation.save_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False).then(
        fn=conversation.clear_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    )
    change.click(
        fn=conversation.set_system_prompt,
        inputs=[system],
        outputs=[system],
        queue=False,
    )
    reset.click(
        fn=conversation.reset_system_prompt,
        inputs=[],
        outputs=[system],
        queue=False,
    )


demo.queue().launch(debug=True)
