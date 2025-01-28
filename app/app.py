from flask import Flask, request, redirect, render_template, url_for, jsonify
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline   # <-- For DeepSeek pipeline
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

# Load the M2M100 model & tokenizer for translation
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

##################################################################
#            CHAT ENDPOINT (Using DeepSeek Pipeline)            #
##################################################################

# 1) Initialize the pipeline
deepseek_pipe = pipeline(
    "text-generation", 
    model="deepseek-ai/DeepSeek-R1", 
    trust_remote_code=True
)

# We'll keep a very simple global in-memory conversation store
conversation_history = []

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    A simple chat endpoint using deepseek-ai/DeepSeek-R1 via pipeline.
    """
    global conversation_history
    error = None

    if request.method == 'POST':
        user_prompt = request.form.get('prompt', '').strip()
        if not user_prompt:
            error = "Please enter a prompt."
        else:
            # 1) Add user's message to conversation
            conversation_history.append({"role": "user", "content": user_prompt})

            # 2) Prepare messages for the pipeline
            #    The pipeline can accept a list of messages if the model supports it
            #    (deepseek-ai/DeepSeek-R1 does support passing messages with roles).
            #    If it only supports raw text, you'll need to convert them into a single string.
            
            # We'll attempt the "messages" format (as shown in the snippet):
            # messages = [
            #   {"role": "user", "content": "Who are you?"}
            # ]
            # Here, we simply pass in the entire conversation to let the model see the context:
            output = deepseek_pipe(conversation_history)

            # 3) The pipeline returns a list. The usual key is 'generated_text'
            if isinstance(output, list) and len(output) > 0:
                model_reply = output[0].get('generated_text', '').strip()
            else:
                model_reply = "No response."

            # 4) Add the model's response to the conversation
            conversation_history.append({"role": "assistant", "content": model_reply})

            return redirect(url_for('chat'))

    # Render the chat template with existing conversation
    return render_template("tikgpt.html", conversation=conversation_history, error=error)

##################################################################
#       OPTIONAL: An API endpoint if you still want one         #
##################################################################

@app.route('/api-chat', methods=['POST'])
def chatapi():
    """
    Endpoint that generates text using the pipeline for JSON requests.
    Expects a JSON body: {"prompt": "..."}
    Returns JSON: {"response": "... model output ..."}
    """
    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # We could just call the pipeline with the prompt:
    messages = [{"role": "user", "content": prompt}]
    output = deepseek_pipe(messages)
    if isinstance(output, list) and len(output) > 0:
        ai_reply = output[0].get('generated_text', '').strip()
    else:
        ai_reply = "No response."

    return jsonify({"response": ai_reply})


if __name__ == '__main__':
    # Make sure there's a 'static' folder to save MP3 files
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
