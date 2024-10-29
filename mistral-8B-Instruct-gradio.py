import os
import time
import spaces
import torch
import gradio as gr
import json

from huggingface_hub import snapshot_download
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

HF_TOKEN = os.environ.get("HF_TOKEN", None)

PLACEHOLDER = """
<center>
<p>Chat with Mistral AI LLM.</p>
</center>
"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
.examples {
    display: None;
}
"""


# download model
mistral_models_path = Path.home().joinpath('mistral_models', '8B-Instruct')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Ministral-8B-Instruct-2410", allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], local_dir=mistral_models_path)

# tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
tekken = tokenizer.instruct_tokenizer.tokenizer
tekken.special_token_policy = SpecialTokenPolicy.IGNORE
model = Transformer.from_folder(
    mistral_models_path,
    device=device,
    dtype=torch.bfloat16)
  
@spaces.GPU()
def stream_chat(
    message: str, 
    history: list,
    tools: str,
    temperature: float = 0.3, 
    max_tokens: int = 1024, 
):
    print(f'message: {message}')
    print(f'history: {history}')

    conversation = []
    for prompt, answer in history:
        conversation.append(UserMessage(content=prompt))
        conversation.append(AssistantMessage(content=answer))
        
    # for item in history:
    #     if item[role] == "user":
    #         conversation.append(UserMessage(content=item[content]))
    #     elif item[role] == "assistant":
    #         conversation.append(AssistantMessage(content=item[content]))
            
    conversation.append(UserMessage(content=message))
    
    print(f'history: {conversation}')

    tools = f'function_params = {{{tools}}}'
    local_namespace = {}
    exec(tools, globals(), local_namespace)
    function_params = local_namespace.get('function_params', {})
    
    completion_request = ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    **function_params
                )
            )
        ] if tools else None,
        messages=conversation)
    
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    
    out_tokens, _ = generate(
        [tokens], 
        model, 
        max_tokens=max_tokens, 
        temperature=temperature,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    
    for i in range(len(result)):
        time.sleep(0.05)
        yield result[: i + 1]
   


tools_schema = """
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location.",
            },
        },
        "required": ["location", "format"],
    },
"""

chatbot = gr.Chatbot(height = 600, placeholder = PLACEHOLDER)
with gr.Blocks(theme="citrus", css=CSS) as demo:
    gr.ChatInterface(
        fn = stream_chat,
        title = "Mistral-lab",
        chatbot = chatbot,
        # type="messages",
        fill_height = True,
        examples = [
            ["Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option."],
            ["What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter."],
            ["Tell me a random fun fact about the Roman Empire."],
            ["Show me a code snippet of a website's sticky header in CSS and JavaScript."],
            ["What is the best possible option for new code gen tools? Exlcuding low code"]
        ],
        cache_examples = False,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
        additional_inputs=[
            gr.Textbox(
                value = tools_schema,
                label = "Tools schema",
                lines = 10,
                render=False,
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.3,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=8192,
                step=1,
                value=1024,
                label="Max new tokens",
                render=False,
            ),
        ],
    )
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")


if __name__ == "__main__":
    demo.launch()
