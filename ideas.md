- radio filters for episode + model type
- use mpt-30b front end + falcon api endpoint code for hf + custom endpoint code for openai
- for timestamps, do separate semantic search on section titles against input and just put see this section for more
- gpt-3.5-turbo-16k and mpt-7b-chat-16k as tester backbones
- follow ask if context is relevant to last request in the message dialogue, if not, delete old context insert new context, keep dialogue history though
- tiktok token counter for left-side truncation 
- semantic search on section summaries, full text as context input context -> works with question callback and 16k context windows, keep one "context" section at a time regardless of dialogue.


