from flask import Flask, request, redirect, render_template, url_for,jsonify
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer,AutoTokenizer, AutoModelForCausalLM
from langdetect import detect, LangDetectException
import sentencepiece  # Needed by M2M100 for tokenization
from gtts import gTTS
import os
import uuid

app = Flask(__name__)

# Map language names to M2M100 language codes
LANGUAGES = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Arabic': 'ar'
}

# Load the M2M100 model & tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def home():
    translation = ""
    error = ""
    source_lang = 'auto'
    target_lang = 'en'
    input_text = ""
    mp3_url = None  # Will hold the path to the generated TTS file

    if request.method == 'POST':
        input_text = request.form.get('text', '')
        source_lang = request.form.get('source_lang', 'auto')
        target_lang = request.form.get('target_lang', 'en')

        if input_text.strip():
            try:
                # Detect language if user chose "auto"
                if source_lang == 'auto':
                    try:
                        detected = detect(input_text)
                        if detected not in LANGUAGES.values():
                            raise ValueError(f"Detected language '{detected}' not supported.")
                        source_lang = detected
                    except LangDetectException:
                        error = "Could not detect the language. Please select manually."
                        source_lang = 'auto'  # keep it

                if not error:
                    # Set source language for tokenizer
                    tokenizer.src_lang = source_lang
                    encoded = tokenizer(input_text, return_tensors="pt")

                    # Generate translation, specifying the target language
                    generated_tokens = model.generate(
                        **encoded,
                        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
                    )
                    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                    # Generate TTS
                    if translation.strip():
                        tts = gTTS(text=translation, lang=target_lang)
                        filename = f"tts_{uuid.uuid4().hex}.mp3"
                        filepath = os.path.join("static", filename)
                        tts.save(filepath)
                        mp3_url = f"/static/{filename}"

            except Exception as e:
                error = f"Translation error: {str(e)}"

    return render_template(
        'index.html', 
        translation=translation,
        error=error,
        languages=LANGUAGES,
        source_lang=source_lang,
        target_lang=target_lang,
        input_text=input_text,
        mp3_url=mp3_url
    )

# Load a GPT-style model from Hugging Face
MODEL_NAME = "facebook/opt-1.3b"  # Or any other causal LM on Hugging Face
tokenizer_gpt = AutoTokenizer.from_pretrained(MODEL_NAME)
model_gpt =  AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)

# We'll keep a very simple global in-memory conversation store (for demonstration):
conversation_history = []

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_history
    error = None

    if request.method == 'POST':
        user_prompt = request.form.get('prompt', '').strip()
        if not user_prompt:
            error = "Please enter a prompt."
        else:
            # 1) Add user's prompt to conversation
            conversation_history.append({"role": "user", "content": user_prompt})

            # 2) Prepare input for GPT
            # A simple approach: join all messages into a single string, 
            # but for real multi-turn chat, you’d want a more robust approach
            full_context = ""
            for msg in conversation_history:
                if msg["role"] == "user":
                    full_context += f"User: {msg['content']}\n"
                else:  # assistant
                    full_context += f"AI: {msg['content']}\n"
            full_context += "AI: "

            # 3) Generate the model output
            inputs = tokenizer_gpt.encode(full_context, return_tensors='pt')
            # Adjust max_length, temperature, top_k, etc. as needed
            outputs = model_gpt.generate(
                inputs, 
                max_length=len(inputs[0]) + 50, 
                num_return_sequences=1,
                do_sample=True,  # For creative generation
                temperature=0.9,
                top_p=0.9,
                pad_token_id=tokenizer_gpt.eos_token_id
            )
            # Decode and extract the new text the model appended
            generated_text = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)

            # We only want the new portion after "AI: "
            # A simplistic approach:
            answer = generated_text.split("AI:")[-1].strip()

            # 4) Add the model's response to the conversation
            conversation_history.append({"role": "assistant", "content": answer})

            return redirect(url_for('chat'))

    # Render the chat template with existing conversation
    return render_template("tikgpt.html", conversation=conversation_history, error=error)

@app.route('/api-chat', methods=['POST'])
def chatapi():
    """
    Endpoint that generates text using GPT-2.
    Expects a JSON body: {"prompt": "..."}
    Returns JSON: {"response": "... GPT-2 output ..."}
    """
    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    # 1) Validate input
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # 2) Encode the prompt
    input_ids = tokenizer_gpt.encode(prompt, return_tensors='pt')
    
    # If you're using GPU, move the input to CUDA:
    # input_ids = input_ids.to("cuda")

    # 3) Generate text
    # Adjust parameters as desired:
    # - max_length: total tokens for input + output
    # - temperature: how "creative" the model is
    # - top_p, top_k, etc. for nucleus or top-k sampling
    output_ids = model_gpt.generate(
        input_ids,
        max_length=len(input_ids[0]) + 50,  # e.g. prompt length + 50 tokens
        num_return_sequences=1,
        do_sample=True,      # for sampling (creative) 
        top_p=0.9,           # nucleus sampling
        temperature=0.9,     # creativity
        # top_k=50,          # if you want top-k
        pad_token_id=tokenizer_gpt.eos_token_id  # avoid errors with GPT-2
    )

    # 4) Decode the generated tokens
    generated_text = tokenizer_gpt.decode(output_ids[0], skip_special_tokens=True)

    # (Optional) If you want to remove the original prompt part from the response,
    # you can do something like:
    # ai_reply = generated_text[len(prompt):].strip()

    ai_reply = generated_text  # The full prompt + continuation

    # 5) Return the AI’s response in JSON
    return jsonify({"response": ai_reply})


if __name__ == '__main__':
    # Make sure there's a 'static' folder to save MP3 files
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
