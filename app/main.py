from flask import Flask, request, render_template_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load Model and Tokenizer with proper configuration
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

@app.route('/', methods=['GET', 'POST'])
def home():
    generated_text = ""
    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
        
        if user_input:
            # Use the user input directly as the prompt
            prompt = user_input
            
            try:
                # Tokenize with attention mask
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Generate text
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                # Decode and extract generated text
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = full_text
                
            except Exception as e:
                generated_text = f"Error: {str(e)}"

        return render_template_string(HTML_TEMPLATE, user_input=user_input, generated_text=generated_text)
    
    return render_template_string(HTML_TEMPLATE, user_input="", generated_text="")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Generator GPT2</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f0f2f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; margin-bottom: 1.5rem; }
        textarea { width: 100%; height: 120px; padding: 1rem; margin: 1rem 0; border: 2px solid #dadce0; border-radius: 5px; font-size: 16px; }
        button { background: #1a73e8; color: white; border: none; padding: 0.8rem 1.5rem; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #1557b0; }
        .output { margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 5px; border: 1px solid #dadce0; }
        .output strong { color: #202124; display: block; margin-bottom: 0.5rem; }
        .output p { color: #3c4043; margin: 0; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Text Generator GPT</h1>
        <form method="POST">
            <textarea name="user_input" placeholder="Enter text to generate...">{{ user_input }}</textarea>
            <button type="submit">Generate</button>
        </form>
        {% if generated_text %}
        <div class="output">
            <strong>Generated Text:</strong>
            <p>{{ generated_text }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)