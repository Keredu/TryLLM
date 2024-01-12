import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "phixtral-2x2_8"
instruction = '''
    def print_prime(n):
        """
        Print all primes between 1 and n
        """
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer without manually moving them to a device
model = AutoModelForCausalLM.from_pretrained(
    f"mlabonne/{model_name}", 
    torch_dtype="auto", 
    load_in_4bit=True, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    f"mlabonne/{model_name}", 
    trust_remote_code=True
)

# Tokenize the input string and move the input tensors to the appropriate device
inputs = tokenizer(
    instruction, 
    return_tensors="pt", 
    return_attention_mask=False
).to(device)  # Move input tensors to the device

# Generate text using the model
outputs = model.generate(**inputs, max_length=200)

# Decode and print the output
text = tokenizer.batch_decode(outputs)[0]
print(text)

