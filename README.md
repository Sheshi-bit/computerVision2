A streamlined Computer Vision + LLM summarization pipeline built with YOLOs object detection and a Groq-powered LLM for result interpretation — wrapped in a Streamlit web app.

This repository demonstrates a real-world CV + NLP integration you can *ship*, *demo*, or *reuse* in production-adjacent workflows.

🚀 Features

🔍 YOLOs Object Detection

  * Uses `hustvl/yolos-tiny` from Hugging Face Transformers.
  * Detects objects with confidence thresholds and draws bounding boxes.

🤖 Groq LLM Summarizer

  * Sends raw detection outputs to a Groq LLM for concise interpretations.
  * Auto-discovers supported Groq models at runtime — no hardcoded deprecated IDs.

🖼️ Streamlit UI

  * Upload images.
  * Run detection.
  * See annotated results and summaries instantly.

📦 Installation

Clone the repo:

```bash
git clone https://github.com/Sheshi-bit/computerVision2.git
cd computerVision2
```

Create a Python environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, use:

```bash
pip install groq transformers torch streamlit pillow
```

---

🛠️ Configuration

 1. Groq API Key

Create an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Or replace in code (temporary only — do NOT commit keys!).

---

💡 How It Works

1. Upload Image
   Via Streamlit UI.

2. Run YOLOs Detection

   * Processes image.
   * Outputs bounding boxes + labels + scores.

3. Summarize with Groq

   * Automatically selects a valid Groq chat model.
   * Sends detection text to LLM for a human-friendly summary.

4. Display Results

   * Render boxes on image.
   * Show the summary.

🚀 Usage

Start the app:

```bash
streamlit run main.py
```

Open your browser to the local Streamlit URL shown in the terminal.

Upload an image (`.jpg`, `.jpeg`, `.png`) and watch the magic.

---

🧪 Example Output

Input:

> `Image of people, chairs, and a table`

Detection:

```
Detected person (confidence=0.95) at ...
Detected chair (confidence=0.92) at ...
Detected table (confidence=0.91) at ...
```

Groq Summary:

> “The image contains two people, multiple chairs, and a table — all confidently identified.”

---

📐 Architecture Overview (Corporate Laydown)

```
STREAMLIT UI
    └▶ Upload Image
           └▶ YOLOs Model (HuggingFace)
               └▶ Object Detection Output
                     └▶ Auto Groq Model Picker
                          └▶ Groq Chat API
                              └▶ Summarization
                                   └▶ UI Display
```

---

⚙️ Dependencies

| Category | Library                                      |
| -------- | -------------------------------------------- |
| CV       | `transformers`, `torch`, `hustvl/yolos-tiny` |
| LLM      | `groq`                                       |
| UI       | `streamlit`, `pillow`                        |

---

💡 What’s Strong About This Repo

✅ Auto model discovery prevents future outages
✅ No hardcoded deprecated IDs — aligns with real infra lifecycle
✅ Combines vision + LLM in a single pipeline
✅ Works locally and can be deployed with minimal tweaks

---

❗ Limitations / Next Actions

⚠ Groq models evolve rapidly

* Your code handles this, but tests for higher-level quality are missing.

⚠ No caching or batching

* Summary requests are synchronous.

⚠ No error handling UI

* If Groq or detection fails, Streamlit crashes.

Future improvements (suggestions):

* Add retry and rate-limit handling for Groq.
* Support multiple images in one session.
* Add deployment config (`Dockerfile`, `Streamlit Cloud`, or Vercel).

---

🧩 Project Structure

```
📁 computerVision2/
├── main.py
├── requirements.txt
├── README.md
└── assets/   (optional test images)
```

---

🧑‍💻 Contributing

This repo is a solo effort, but you can contribute:

* Fix bugs
* Improve UX
* Add tests
* Containerize

Submit PRs, not excuses.

---

📝 License

MIT © 2026 — You include proper credit if reused publicly.
No monkeys, no nonsense.

---

📞 Contact

Maintained by Sheshi-bit
GitHub: `github.com/Sheshi-bit`

---

If you want, I can also generate a requirements.txt, Dockerfile, and deployment guide (Streamlit Cloud / Heroku / Railway).
