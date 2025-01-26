from flask import Flask, request, render_template_string
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

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>TikTranslate Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --neon-pink: #ff007f;
            --neon-blue: #00f3ff;
            --bg-dark: #0f0f0f;
            --glass-bg: rgba(255,255,255,0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'SF Pro Display', system-ui, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #1a1a1a, #0a0a0a);
            min-height: 100vh;
            color: white;
            padding: 1rem;
        }

        .container {
            max-width: 800px;
            margin: 2rem auto;
            animation: fadeIn 0.5s ease-in;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
        }

        .header h1 {
            font-size: 2.8rem;
            background: linear-gradient(45deg, var(--neon-pink), var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 25px rgba(255,0,127,0.3);
            margin-bottom: 0.5rem;
        }

        .lang-selectors {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        select {
            background: var(--glass-bg);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 0.8rem;
            color: white;
            font-size: 1rem;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        }

        select:hover {
            border-color: var(--neon-pink);
            box-shadow: 0 0 15px rgba(255,0,127,0.2);
        }

        .translator-box {
            background: var(--glass-bg);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        textarea {
            width: 100%;
            height: 150px;
            background: rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1rem;
            color: white;
            font-size: 1.1rem;
            resize: none;
            transition: all 0.3s ease;
            margin: 1rem 0;
        }

        textarea:focus {
            outline: none;
            border-color: var(--neon-blue);
            box-shadow: 0 0 20px rgba(0,243,255,0.2);
        }

        .translate-btn {
            background: linear-gradient(45deg, var(--neon-pink), var(--neon-blue));
            border: none;
            padding: 1rem 2rem;
            border-radius: 15px;
            color: white;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            width: 100%;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .translate-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(255,0,127,0.4);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(255,255,255,0.02);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.08);
            animation: slideUp 0.4s ease-out;
        }

        .result-text {
            font-size: 1.2rem;
            line-height: 1.6;
            color: #fff;
            opacity: 0.9;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .error-box {
            color: var(--neon-pink);
            margin-top: 1rem;
            padding: 1rem;
            border: 1px solid var(--neon-pink);
            border-radius: 10px;
            background: rgba(255,0,127,0.1);
        }

        audio {
            margin-top: 1rem;
            width: 100%;
            outline: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TikTranslate Pro</h1>
            <p>🌍 AI-Powered Multilingual Translation</p>
        </div>

        <form method="POST">
            <div class="lang-selectors">
                <select name="source_lang">
                    <option value="auto" {% if source_lang == 'auto' %}selected{% endif %}>Detect Language</option>
                    {% for lang, code in languages.items() %}
                    <option value="{{ code }}" {% if source_lang == code %}selected{% endif %}>{{ lang }}</option>
                    {% endfor %}
                </select>

                <select name="target_lang">
                    {% for lang, code in languages.items() %}
                    <option value="{{ code }}" {% if target_lang == code %}selected{% endif %}>{{ lang }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="translator-box">
                <textarea name="text" placeholder="Type or paste your text here...">{{ input_text }}</textarea>
                <button type="submit" class="translate-btn">🚀 Translate Now</button>

                {% if translation %}
                <div class="result-box">
                    <div class="result-text">{{ translation }}</div>

                    {% if mp3_url %}
                        <!-- Audio player for TTS -->
                        <audio controls>
                            <source src="{{ mp3_url }}" type="audio/mpeg">
                            Your browser does not support the audio element.
                        </audio>
                    {% endif %}
                </div>
                {% endif %}

                {% if error %}
                <div class="error-box">
                    {{ error }}
                </div>
                {% endif %}
            </div>
        </form>
    </div>

    <script>
        // Auto-resize textarea
        const textarea = document.querySelector('textarea');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });

        // Initial resize
        textarea.dispatchEvent(new Event('input'));
    </script>
</body>
</html>
"""

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
                # If user chose "Detect Language", we actually detect
                if source_lang == 'auto':
                    try:
                        detected = detect(input_text)
                        # Check if detected language is in our LANGUAGES codes
                        if detected not in LANGUAGES.values():
                            raise ValueError(f"Detected language '{detected}' not supported.")
                        source_lang = detected
                    except LangDetectException:
                        error = "Could not detect the language. Please select manually."
                        source_lang = 'auto'  # keep it

                # Proceed if no detection error
                if not error:
                    # Tell the tokenizer what the source language is
                    tokenizer.src_lang = source_lang

                    # Encode the text, then force the target language in generation
                    encoded = tokenizer(input_text, return_tensors="pt")
                    generated_tokens = model.generate(
                        **encoded,
                        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
                    )
                    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                    # ========== ADD TTS HERE ==========
                    # If we have a non-empty translation, generate TTS
                    if translation.strip():
                        from gtts import gTTS

                        tts = gTTS(text=translation, lang=target_lang)
                        # Generate a unique filename to avoid overwriting
                        filename = f"tts_{uuid.uuid4().hex}.mp3"
                        filepath = os.path.join("static", filename)

                        # Save the MP3 into ./static
                        tts.save(filepath)
                        # We'll pass this path to our template
                        mp3_url = f"/static/{filename}"

            except Exception as e:
                error = f"Translation error: {str(e)}"

    return render_template_string(
        HTML,
        translation=translation,
        error=error,
        languages=LANGUAGES,
        source_lang=source_lang,
        target_lang=target_lang,
        input_text=input_text,
        mp3_url=mp3_url
    )

if __name__ == '__main__':
    # Make sure there's a 'static' folder to save the MP3 files
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
