from pathlib import Path
import pandas as pd
import numpy as np
import gradio as gr
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

# --- PATH & DATA SETUP ---
# If deploying on Hugging Face Spaces, place 'df_rag_block.parquet' and 'review_index_*.faiss' 
# directly in the same root folder as app.py.
DATA_DIR = Path(".")

# Load the dataframe using Parquet for fast, efficient loading
csv_path = DATA_DIR / "df_rag_block.csv"
df_rag = pd.read_csv(csv_path)
print(f"DataFrame successfully loaded from csv! Shape: {df_rag.shape}")

# Determine device (CPU vs GPU)
if torch.cuda.device_count() > 0:
    my_device = "cuda"
    print(f"You have {torch.cuda.device_count()} GPUs available.")
else:
    my_device = "cpu"
    print("You have no GPUs available. Running on CPU.")

# --- LOAD MODELS & VECTOR INDEX ---
embeddings_model_name = 'BAAI/bge-small-en-v1.5'
embeddings_model = SentenceTransformer(embeddings_model_name, cache_folder="./HF-CACHE", device=my_device)

# Load FAISS index
safe_model_name = embeddings_model_name.replace("/", "_")
file_name_index = f"review_index_{safe_model_name}.faiss"
index_path = DATA_DIR / file_name_index
index = faiss.read_index(str(index_path))
print(f"Successfully loaded index with {index.ntotal} vectors.")

# Load LLM Decoder Model (Qwen)
llm_model_name = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir="./HF-CACHE")
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    cache_dir="./HF-CACHE",
    device_map="auto",
    torch_dtype="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# --- RAG & CHAT LOGIC ---
def get_context(query, embeddings_model, index, df, k):
    query_vector = embeddings_model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector, k)
    retrieved_texts = df_rag.iloc[indices[0]]['rag_block'].unique().tolist()
    return retrieved_texts

def chat(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    gen_config = GenerationConfig(
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = qwen_pipeline(messages, generation_config=gen_config)
    return outputs[0]["generated_text"][-1]["content"]

def format_chat_prompt(message, chat_history, max_convo_length=10):
    prompt = ""
    for turn in chat_history[-max_convo_length:]:
        role = turn.get("role")
        content = turn.get("content")
        speaker = "User" if role == "user" else "Assistant"
        prompt = f"{prompt}\n{speaker}: {content}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:
    return prompt

def respond(message, chat_history, k_val, temp_val):
    context_list = get_context(message, embeddings_model, index, df_rag, k=int(k_val))
    context_list = list(dict.fromkeys(context_list))
    context_str = "\n\n".join(context_list)

    formatted_prompt = format_chat_prompt(message, chat_history)

    system_instruction = (
        "You are a medical research assistant. "
        "Use the provided <context> (which includes patient reviews and ratings such as EaseofUse, Effectiveness and Satisfaction) to answer the <question>. "
        "Prioritize answers with larger UsefulCount. "
        "Do not make up medical facts outside of the provided context. "
        "Do not use outside knowledge.\n\n"
        "Instructions:\n"
        "- If the question asks about 'side effects', prioritize the 'Side Effects' field.\n"
        "- If the question asks about 'effectiveness' or comparative drug performance, prioritize the numerical 'Effectiveness' ratings over subjective text sentiment.\n"
        "- If the question asks about 'satisfaction', prioritize the numerical 'Satisfaction' ratings over subjective text sentiment.\n"
        "- If the question asks about 'ease of use', prioritize the numerical 'EaseofUse' ratings and specific usability comments in the reviews.\n"
        "- When performing drug comparisons, prioritize numerical ratings across the metadata rather than selective review quotes.\n"
        "- When summarizing data or trends from the snippets, synthesize the patterns directly from the retrieved text rather than refusing due to a lack of global data.\n"
        "- Do not invent rating scales (like 6-10) if the ratings are out of 5.\n"
        "- When answering, directly address the user's prompt using the patterns, counts, and examples found within the retrieved text snippets.\n"
        "- If a specific data point is missing from the snippets, concisely state what the retrieved snippets show rather than issuing a blanket refusal.\n"
        "- Answer directly and concisely based on the text provided.\n"
        "- If a comparative question is asked (e.g., comparing age groups) and the retrieved context shows identical standardized metadata for both groups, explicitly state that the context shows no difference rather than forcing a distinction."
    )

    user_prompt_with_rag = f"Context:\n{context_str}\n\nConversation/Question:\n{formatted_prompt}"

    bot_message = chat(system_prompt=system_instruction, user_prompt=user_prompt_with_rag)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return "", chat_history, context_str


# --- GRADIO CUSTOM CSS & UI ---
custom_css = """
.webmd-header-box {
    background-color: #0070AF;
    padding: 20px 25px;
    border-radius: 10px;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    width: 100%;
}
.webmd-header-box h1, .webmd-header-box p, .webmd-header-box * {
    color: white !important;
}
.shadow-divider {
    border: none;
    height: 4px;
    background: linear-gradient(to bottom, rgba(0,0,0,0.1), rgba(0,0,0,0.02));
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
    margin: 15px 0;
}
button.primary {
    background-color: #3B95D1 !important;
    border-color: #3B95D1 !important;
    color: white !important;
}
button.primary:hover {
    background-color: #2E81B8 !important;
}
input[type="range"] {
    accent-color: #6366f1 !important;
}
input[type="range"]::-webkit-slider-thumb {
    background-color: #6366f1 !important;
    border: 2px solid #6366f1 !important;
}
input[type="range"]::-moz-range-thumb {
    background-color: #6366f1 !important;
    border: 2px solid #6366f1 !important;
}
input[type="range"]::-ms-thumb {
    background-color: #6366f1 !important;
    border: 2px solid #6366f1 !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
    with gr.Row():
        gr.Markdown(
            """
            <div class="webmd-header-box">
                <h1 style="margin: 0; font-size: 26px;">💊 Medical Drug Review Assistant</h1>
                <p style="margin: 8px 0 0 0; font-size: 15px; opacity: 0.95;">
                    &nbsp;&nbsp;&nbsp;&nbsp;<em>An NLP application for analyzing WebMD patient reviews</em>
                </p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400, show_label=False)
            gr.HTML('<div class="shadow-divider"></div>')
            msg = gr.Textbox(
                label="Query",
                placeholder="Ask a medical question (e.g., What are the side effects of Lisinopril?)...",
                container=True
            )
            with gr.Column():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.ClearButton(components=[msg, chatbot], value="Clear Conversation")

            gr.Examples(
                examples=[
                    "What are the reported side effects for Lisinopril?",
    "What is the average satisfaction rating for patients using Lisinopril for High Blood Pressure?",
    "Which drug is generally considered more effective for Depression: Cymbalta or Lexapro?",
    "Do older patients (75 or over) report different side effects for Lisinopril than younger patients (25-34)?",
    "Can you summarize the patient experience for Metformin regarding Type 2 Diabetes?",
    "Why might a patient give Metformin a low satisfaction score (1/5)?"
                ],
                inputs=msg
            )

        with gr.Column(scale=1):
            gr.Markdown("## 📊 Project Dashboard")
            with gr.Accordion("⚙️ Model Controls", open=True):
                slider_k = gr.Slider(minimum=5, maximum=40, value=30, step=5, label="Most Similar Reviews")
                slider_temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Answering Creativity")
            
            with gr.Accordion("🔍 Retrieved Context Inspector", open=False):
                context_output = gr.Textbox(label="Most Relevant Documents Extracted", lines=10, interactive=False)

    btn.click(
        respond, 
        inputs=[msg, chatbot, slider_k, slider_temp], 
        outputs=[msg, chatbot, context_output]
    )
    msg.submit(
        respond, 
        inputs=[msg, chatbot, slider_k, slider_temp], 
        outputs=[msg, chatbot, context_output]
    )

gr.close_all()
demo.launch(
    theme=custom_theme, 
    css=custom_css, 
    ssr_mode=False, 
    debug=True
)