from flask import Flask, request, jsonify
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN"  # Replace with your Hugging Face Token
MODEL_NAME = "meta-llama/Llama-3.3b-hf"  # Change this to your specific model
SERPAPI_KEY = "YOUR_SERPAPI_KEY"  # Replace with your SerpAPI Key

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
model.eval()

# Initialize Flask App
app = Flask(__name__)

# Helper Function: Query SerpAPI
def query_serpapi(query):
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    response = requests.get(url)
    return response.json()

# Helper Function: Process with Llama
def llama_process(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# LangChain Setup
prompt_template = PromptTemplate.from_template(
    template="Based on the information: {context}, generate a detailed explanation."
)
llm_chain = LLMChain(prompt=prompt_template, llm=llama_process)

# Route: Handle Request from FlutterFlow
@app.route('/process', methods=['POST'])
def process_request():
    data = request.json
    user_input = data.get('input', '')

    # Step 1: Query SerpAPI
    serp_response = query_serpapi(user_input)
    context = serp_response.get('organic_results', [{}])[0].get('snippet', '')

    # Step 2: Process with Llama using LangChain
    final_output = llm_chain.run(context=context)

    return jsonify({'response': final_output})

# Main: Run Flask App
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

