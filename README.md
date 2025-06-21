# 🗣️ AI Persona Matcher

**Let your AI twin break the ice.**  
Upload (or paste) two ChatGPT conversation snippets → get concise personality profiles, a playful 12-line dialogue, and a compatibility score (0–100).

[![Streamlit demo](https://img.shields.io/badge/demo-live-brightgreen)](https://<your-streamlit-url>)
![MIT](https://img.shields.io/badge/License-MIT-blue)

---

## ✨ Live demo

> **Try it in your browser (no install):**  
> **👉 <https://your-streamlit-url>**

![demo gif](docs/peek.gif)

---

## Quick start (local)

```bash
git clone https://github.com/<you>/persona-matcher.git
cd persona-matcher
pip install -r requirements.txt
# add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run app.py
