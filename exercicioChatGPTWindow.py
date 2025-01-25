from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Carregar Modelo e Tokenizador
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Definir Prompt com Exemplo de In-Context Learning
prompt = """
Translate the next statements to Portugues:

Inglês: "Hello, how are you?"
Português: ""
"""

# 3. Tokenização e Geração
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=300,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1
)

# 4. Decodificação e Saída
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("="*50 + "\nTexto Gerado:\n" + "="*50)
print(generated_text)
