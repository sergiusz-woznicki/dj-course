import mlflow
from openai import OpenAI
import os

# --- OpenAI Client Configuration (for local server communication) ---
# An API key is not strictly needed, but the client requires a Base URL to be passed
# The Base URL points to the locally running Llama.cpp server

# LLAMA_CPP_SERVER = {
#     "engine": "llama-cpp",
#     "model": "whatever-model", # 🔥🔥🔥 tu nie ma różnicy co wpiszesz, bo na tym URLu jest tylko 1 model
#     "base_url": "http://localhost:8080/v1",
# }
OLLAMA_SERVER = {
    "engine": "ollama",
    "model": "hf.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M",
    # "model": "hf.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
    "model": "hf.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",
    # "model": "hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M",
    # "model": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
    # "model": "hf.co/bartowski/microsoft_Phi-4-mini-reasoning-GGUF:Q4_K_M",
    "base_url": "http://localhost:11434/v1",
}
SERVER = OLLAMA_SERVER

print(f"Using {SERVER['engine']} server at {SERVER['base_url']} with model {SERVER['model']}")

# --- MLflow Configuration ---
# Enable automatic logging for OpenAI calls
# This will work because the local server is compatible with the OpenAI API
mlflow.openai.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# # Set the MLflow experiment
mlflow.set_experiment(f"DJ_local_model_tracking_{SERVER['engine']}")

client = OpenAI(
    base_url=SERVER["base_url"],
    api_key="sk-not-needed",
)

messages=[
    {"role": "system", "content": "You are a happy AI assistant, eager to help. No bullshit, just code."},
    {"role": "user", "content": "Write a short but fancy example using lists and maps in python. One liner, no more lines of code, just ONE! No comments, just code."},
]

# --- Perform Inference and Tracking ---
with mlflow.start_run() as run:
    print(f"Tracking started in MLflow Run ID: {run.info.run_id}")
 
    # Model call
    try:
        completion = client.chat.completions.create(
            model=SERVER["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )

        # Retrieve and display the response
        response_text = completion.choices[0].message.content
        print("\n--- Model Response ---")
        print(response_text)
        print("------------------------")

    except Exception as e:
        print(f"Error connecting to the model: {e}")
        print(f"Make sure the {SERVER['engine']}.server is running at {SERVER['base_url']}.")

    # The interaction data (prompt, response, parameters, tokens) is now
    # automatically logged as a 'Trace' in MLflow thanks to autologging.
    print(f"\nThe Trace is available in the MLflow UI in the 'Traces' section for Run ID: {run.info.run_id}")
