import os
import logging
import gradio as gr

from src.chat_class import Chat


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.warning("READY. App started...")


EPISODES = [
    "Jun 30, 2023: Wagner rebels, SCOTUS ends AA, AI M&A, startups gone bad, spacetime warps & more (E135)",
    "Jun 23, 2023: Ukraine counteroffensive, China tensions, COVID Patient Zero, RFK Jr reaction & more (E134)",
]


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown(
        """<h1><center>Chat with the "All In" Podcast</center></h1>

        A chatbot that knows up-to-date M&A news from the "[All In](https://www.youtube.com/channel/UCESLZhusAkFfsNsApnjF_Cg)" podcast. Start by entering your OpenAI key and selecting an episode of interest 🚀.

"""
    )

    conversation = Chat()
    with gr.Row():
        openai_key = gr.Textbox(
            label="OpenAI Key",
            value="",
            type="password",
            placeholder="sk..",
            info="You have to provide your own OpenAI API key.",
        )
    with gr.Row():
        select_episode = gr.Dropdown(
            EPISODES,
            label="Select Episode",
            info="Will add more episodes later!",
        )
    chatbot = gr.Chatbot().style(height=400)
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
            # with gr.Row():
            # save_history = gr.Button("Cache Ideal Conversation History")

    with gr.Row():
        gr.Markdown(
            'Disclaimer: The "Chat-All-In" application can produce factually incorrect outputs '
            "and should not be solely relied on to produce factually accurate information. While "
            "context retrieval is used to mitigate errors, this method can itself lead to problems "
            "for edge cases.",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot, openai_key, select_episode],
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
        inputs=[system, chatbot, openai_key, select_episode],
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
    # save_history.click(
    # fn=conversation.save_history,
    # inputs=[chatbot],
    # outputs=[chatbot],
    # queue=False,
    # )
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
