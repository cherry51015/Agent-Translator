# Agent-Translator  

**Agent-Translator** is an AI-powered document translation system that supports multiple languages and formats like DOCX, PDF, and TXT.  
Built with Python and NLP techniques, it delivers fast, context-aware translations for academic and professional use.  

---

## 🚀 Installation  

Clone the repository:  
```bash
git clone https://github.com/cherry51015/Agent-Translator.git
cd Agent-Translator
Install dependencies:

bash
Copy code
pip install -r agent_translator/requirements.txt
Set your API key (Google Gemini):

Windows (PowerShell):

bash
Copy code
set GEMINI_API_KEY=your_actual_api_key_here
Linux/Mac (bash/zsh):

bash
Copy code
export GEMINI_API_KEY=your_actual_api_key_here
Or create a .env file in the project root:

ini
Copy code
GEMINI_API_KEY=your_actual_api_key_here
▶️ Usage
Translate with the main CLI:

bash
Copy code
python agent_translator/main.py translate academic_paper.docx --target-language "Spanish"
Use the direct translator (simpler):

bash
Copy code
python agent_translator/direct_translator.py academic_paper.docx -t Spanish
Use the resumable translator (batching & resume support):

bash
Copy code
python agent_translator/resumable_translator.py academic_paper.docx -t Spanish --batch-size 3
Run diagnostics:

bash
Copy code
python agent_translator/diagnose.py
⚡ Troubleshooting
1. Missing crew.py or import errors
Check your folder structure (see 📂 Project Structure below).

2. Direct Translator error: input_file required
Run with both the file and target language:

bash
Copy code
python direct_translator.py academic_paper.docx -t Spanish
3. API quota limit reached
If you see:

json
Copy code
"message": "You exceeded your current quota"
It means you hit the 50 free daily Gemini requests.

✅ Fixes:

Wait 24 hours (quota resets daily).

Use batching with the resumable translator:

bash
Copy code
python resumable_translator.py academic_paper.docx -t Spanish --batch-size 3
Upgrade to a paid Gemini plan for more requests.

4. Debugging with logs
Check the detailed error log:

bash
Copy code
cat translation_system.log
📂 Project Structure
bash
Copy code
agent_translator/
 ├── main.py                  # CLI entry point
 ├── direct_translator.py     # Simple direct translation
 ├── resumable_translator.py  # Resume-friendly translator
 ├── diagnose.py              # Diagnostics
 ├── requirements.txt
 ├── config.yaml
 └── src/agent_translator/
      ├── crew.py
      ├── tools/
      └── config/
