"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from model import GPT2PPL
from collections import OrderedDict
import json

# initialize the model
model = GPT2PPL()

print("Please enter your sentence: (Press Enter twice to start processing)")
contents = []
while True:
    line = input()
    if len(line) == 0:
        break
    contents.append(line)
sentence = "\n".join(contents)

result = model(sentence)

# Convert the result to a JSON object
json_result = {
    "metrics": dict(result[0]),
    "conclusion": result[1]
}

# Print the JSON object
print("JSON result:")
print(json.dumps(json_result, indent=2))
