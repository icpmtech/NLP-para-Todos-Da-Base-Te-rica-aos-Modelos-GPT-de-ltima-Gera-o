from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Carrega o modelo e o tokenizador GPT-2 (versão pequena)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prompt inicial
prompt = "The future of AI in World is"

# Tokeniza e converte em tensores
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Gera texto
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=300,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True
)

# Decodifica a sequência gerada
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)
