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
- Stay in character and do not shift into modern AI commentary unless relevant to character’s history.
- Do not provide unsafe technical guidance.

GOAL:
Educate, inspire, and inform users about mathematics, space exploration, perseverance, and the historical context of Katherine Johnson’s contributions.
"""

def respond(message, history, max_tokens, temperature, top_p):
    import os

    client = InferenceClient(
        model="openai/gpt-oss-20b",
        token=os.environ["HF_TOKEN"],
    )

    # Start conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Rebuild conversation history safely
    for turn in history:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant":
            messages.append({"role": "assistant", "content": turn["content"]})

    # Add latest user input
    messages.append({"role": "user", "content": message})

    response = ""

    # Stream response
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            response += token
            yield response


chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    gr.Markdown(
        "This is a historically grounded educational simulation of Katherine Johnson. "
        "It is not the real person."
    )
    chatbot.render()

if __name__ == "__main__":
    demo.launch()
   