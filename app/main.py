from flask import Flask, request, render_template
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
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

if __name__ == '__main__':
    # Make sure there's a 'static' folder to save MP3 files
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
