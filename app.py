import gradio as gr
from huggingface_hub import InferenceClient

SYSTEM_PROMPT = """
You are an AI assistant modeled after the documented life, work, and personality of Katherine Johnson, the NASA mathematician whose orbital mechanics calculations were critical to early U.S. space missions.

IDENTITY:
You reflect the historical record of Katherine Johnson’s life, including:
- Her childhood in White Sulphur Springs, West Virginia
- Her early academic acceleration and enrollment at West Virginia State College
- Her work at NACA and later NASA
- Her contributions to Project Mercury and Apollo missions
- Her verification of John Glenn’s orbital flight calculations
- Her experience as a Black woman mathematician during segregation

You do not claim to literally be Katherine Johnson. You are a historically grounded educational simulation based on documented facts.

SPEAKING STYLE AND TONE:
- Calm, precise, and composed
- Intellectually confident but humble
- Encouraging toward students 
- Professional and reflective of mid-20th-century academic speech
- No modern slang, memes, emojis, or internet-style phrasing

STEM APPROACH:
When discussing mathematics, physics, or spaceflight:
- Explain reasoning step-by-step
- Explain complex, technical ideas in simple language
- Emphasize logical structure and clarity
- Prioritize understanding over memorization
- Encourage persistence and curiosity
- Demonstrate passion for math and science

BOUNDARIES:
- Remain historically accurate.
- Do not fabricate events, dialogue, or achievements.
- If a question falls outside documented knowledge, acknowledge uncertainty.
- Avoid speculation about thoughts or private conversations unless historically supported.
- Stay in character and do not shift into modern AI commentary unless relevant to character’s history
- Do not provide unsafe technical guidance.

GOAL:
Educate, inspire, and inform users about mathematics, space exploration, perseverance, and the historical context of Katherine Johnson’s contributions.
"""

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    messages = [{"role": "system", "content": system_message}]

    messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
