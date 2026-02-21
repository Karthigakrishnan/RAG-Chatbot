# 🌿 DocMind — RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents and ask questions about them. Powered by **Google Gemini** (or **Groq**) and **FAISS** for semantic search.

---

## ✨ Features

- 📄 Upload **PDFs, DOCX, PPTX, XLSX, CSV, HTML, TXT, SQL** files
- 🔍 Semantic search using **FAISS** + **Sentence Transformers**
- 🤖 Answers grounded strictly in your documents — no hallucination
- 🧠 Multi-turn conversation memory
- ⚡ Two backends: **Gemini** (`Main.py`) and **Groq** (`app.py`)
- 🎨 Clean, modern Streamlit UI

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Karthigakrishnan/RAG-Chatbot.git
cd RAG-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Copy `.env.example` to `.env` and fill in your keys:
```bash
copy .env.example .env
```
Then edit `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/).

### 4. Run the app

**Gemini version:**
```bash
streamlit run Main.py
```

**Groq version:**
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
RAG-Chatbot/
├── Main.py            # Gemini-powered RAG app
├── app.py             # Groq-powered RAG app
├── requirements.txt   # Python dependencies
├── .env.example       # API key template (copy to .env)
└── .gitignore
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| LLM (option 1) | Google Gemini (`google-genai`) |
| LLM (option 2) | Groq (`groq`) |
| Embeddings | Sentence Transformers |
| Vector Store | FAISS |
| File Parsing | PyMuPDF, pdfplumber, python-docx, python-pptx |

---

## ⚠️ Notes

- Your `.env` file is **gitignored** — your API keys are never committed.
- The chatbot only answers from uploaded documents. If the answer isn't in the document, it will say so.

---

## 📄 License

MIT License
