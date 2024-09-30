"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from model import GPT2PPL
import json

# initialize the model
model = GPT2PPL(device="cuda")  # Explicitly set to use CUDA

def run_inference(sentence):
    if sentence:
        result = model(sentence)
        
        # Convert the result to a JSON object
        json_result = {
            "metrics": dict(result[0]),
            "conclusion": result[1]
        }
        
        return json_result
    else:
        return {"error": "No input provided."}

# Remove the interactive prompt code
