from flask import Flask, request, render_template_string
from transformers import MarianMTModel, MarianTokenizer
from time import sleep

app = Flask(__name__)

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            try:
                batch = tokenizer([text], return_tensors="pt")
                generated_ids = model.generate(**batch)
                translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                translation = f"🚨 Error: {str(e)}"
            return render_template_string(HTML, translation=translation, original=text)
        return render_template_string(HTML, error="Please enter some text!")
    return render_template_string(HTML)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>TikTranslate</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --neon-pink: #ff007f;
            --neon-blue: #00f3ff;
            --bg-dark: #0f0f0f;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #1a1a1a, #0a0a0a);
            min-height: 100vh;
            color: white;
            padding: 1rem;
        }

        .container {
            max-width: 600px;
            margin: 2rem auto;
            animation: fadeIn 0.5s ease-in;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(45deg, var(--neon-pink), var(--neon-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 15px rgba(255,0,127,0.3);
        }

        .translator-box {
            background: rgba(255,255,255,0.05);
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
            border-radius: 12px;
            padding: 1rem;
            color: white;
            font-size: 1.1rem;
            resize: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            outline: none;
            border-color: var(--neon-pink);
            box-shadow: 0 0 15px rgba(255,0,127,0.2);
        }

        .button-container {
            margin: 1.5rem 0;
            text-align: center;
        }

        .translate-btn {
            background: linear-gradient(45deg, var(--neon-pink), var(--neon-blue));
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 30px;
            color: white;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .translate-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px rgba(255,0,127,0.4);
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
            line-height: 1.5;
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

        .loading {
            display: none;
            text-align: center;
            margin: 2rem 0;
            color: var(--neon-blue);
            font-size: 1.2rem;
        }

        .pulse {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TikTranslate</h1>
            <p>🇺🇸 → 🇫🇷 Instant Social Media Translations</p>
        </div>

        <form method="POST" onsubmit="showLoading()">
            <div class="translator-box">
                <textarea name="text" placeholder="Type your English text here..." 
                          id="inputText">{{ original or '' }}</textarea>
                
                <div class="button-container">
                    <button type="submit" class="translate-btn">
                        <span class="pulse">✨ Translate Now</span>
                    </button>
                </div>

                {% if translation %}
                <div class="result-box">
                    <div class="result-text">{{ translation }}</div>
                </div>
                {% endif %}

                {% if error %}
                <div class="result-box" style="color: var(--neon-pink);">
                    {{ error }}
                </div>
                {% endif %}
            </div>
        </form>
    </div>

    <script>
        function showLoading() {
            const btn = document.querySelector('.translate-btn');
            btn.innerHTML = '⏳ Translating...';
            btn.style.opacity = '0.7';
            btn.disabled = true;
        }

        // Auto-resize textarea
        const textarea = document.getElementById('inputText');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });

        // Add animation on load
        document.addEventListener('DOMContentLoaded', () => {
            document.querySelector('.container').style.opacity = '1';
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)