# FinPilot: GenAI-Powered Financial Advisor

FinPilot is a local web application that helps you understand your income, expenses, savings and budgets, and lets you ask finance questions to an AI assistant. It demonstrates **Generative AI, prompt engineering, embeddings, vector search and Retrieval-Augmented Generation (RAG)** in one beginner-friendly project (built for an MCA Data Science course).

It runs on your own Windows 10/11 computer. Nothing is deployed to the cloud.

> **Educational project.** FinPilot is not a licensed financial advisor. AI answers are general educational guidance, not individualised professional advice, and are never guaranteed to be correct.

---

## 1. Main features

| Page | What you can do |
|---|---|
| **Financial Dashboard** | Upload a transaction CSV (or load the fictional sample), validate it, preview it, see KPIs (total income, expenses, net savings, savings rate, average monthly expenses, highest spending category), view 4 interactive Plotly charts, set monthly budgets with overspending warnings, and download the analysed data. |
| **AI Financial Advisor** | Chat with a Gemini-powered assistant. Python functions calculate every number; the model only explains them. Includes savings-goal feasibility, labelled hypothetical estimates, conversation history, a reset button, and a panel that shows exactly what data will be sent to Gemini. |
| **RAG Knowledge Base** | Upload PDF/TXT documents, chunk them, embed them with Sentence Transformers, store them in a local FAISS index, retrieve the best passages for a question, and get a Gemini answer grounded in those passages with source citations (file name and page). Includes a small evaluation tab. |

Other highlights: Indian Rupee formatting (₹1,23,456), sample data included, works without an API key (dashboard, document indexing and retrieval inspection still work), clear error messages for missing keys, quota limits and network failures.

## 2. Architecture and RAG workflow

```
                 ┌──────────────────────────── Streamlit UI (app.py) ───────────────────────────┐
                 │  Dashboard           AI Advisor                 RAG Knowledge Base           │
                 └────┬──────────────────┬──────────────────────────────┬───────────────────────┘
        data_loader.py│        financial_calculations.py       document_processor.py
        charts.py     │        budget_analyzer.py              vector_store.py (FAISS)
                      ▼                  │                     rag_pipeline.py
                 Pandas / Plotly         ▼                              │
                                  prompts.py ──► gemini_client.py ◄─────┘
                                                 (only text sent to Google)
```

**RAG pipeline (no LangChain / LangGraph):**

1. **Upload and extract** text from PDF (PyMuPDF, page by page) or TXT files.
2. **Chunk** each page into pieces with a configurable chunk size and overlap (page numbers are kept).
3. **Embed** each chunk with `sentence-transformers/all-MiniLM-L6-v2` (downloaded once, on first use).
4. **Store** the vectors in a local **FAISS** index.
5. **Save** the chunk text and source metadata locally (`knowledge_base/vector_store/`), so you do not need to re-index after a restart.
6. **Embed the question** with the same model.
7. **Retrieve** the top-k most similar chunks (cosine similarity).
8. **Generate**: send only the question and the retrieved chunks (never whole PDFs) to Gemini.
9. **Ground** the answer: the prompt tells Gemini to answer only from the numbered passages and to say so when evidence is insufficient.
10. **Cite**: `[1]`, `[2]` markers in the answer are mapped back to the real retrieved chunks (file name and page). Markers that do not match a retrieved chunk are ignored and flagged. If no chunk is relevant enough, Gemini is **not called at all**.

**How accuracy is protected:** all financial figures come from tested Python functions and are passed to the model as "CALCULATED FIGURES". Assumption-based numbers are labelled "HYPOTHETICAL ESTIMATES".

## 3. Technologies

Python 3.11 · Streamlit · Pandas · NumPy · Plotly · Google Gen AI SDK (`google-genai`) · PyMuPDF · Sentence Transformers · FAISS (CPU) · python-dotenv · Pytest

## 4. Project structure

```
FinPilot/
├── app.py                     Streamlit entry point and navigation
├── config.py                  Settings read from environment variables
├── requirements.txt
├── .env.example               Template for your API key (copy to .env)
├── .gitignore
├── pytest.ini
├── run_windows.bat            Double-click to start the app
├── run_tests.bat              Double-click to run the tests
├── .streamlit/config.toml     Theme and privacy-friendly Streamlit settings
├── data/
│   └── sample_transactions.csv        138 fictional transactions (Apr-Sep 2026)
├── knowledge_base/
│   ├── documents/                     5 original sample finance documents (TXT)
│   └── vector_store/                  Generated FAISS index + metadata (git-ignored)
├── src/
│   ├── data_loader.py                 CSV loading and validation
│   ├── financial_calculations.py      Metrics, goals, projections, ₹ formatting
│   ├── budget_analyzer.py             Budget vs actual
│   ├── charts.py                      Plotly figures
│   ├── prompts.py                     System prompt and prompt templates
│   ├── gemini_client.py               Gemini API wrapper with friendly errors
│   ├── document_processor.py          PDF/TXT extraction and chunking
│   ├── vector_store.py                Embedding model + FAISS store (saved locally)
│   ├── rag_pipeline.py                Ingest, retrieve, generate, citations
│   ├── rag_evaluation.py              Small retrieval evaluation helpers
│   ├── dashboard.py                   Page 1
│   ├── advisor.py                     Page 2
│   └── knowledge_base_page.py         Page 3
└── tests/                             Automated tests (no API key needed)
```

## 5. Installation (Windows 10/11)

### 5.1 Install Python 3.11
Download Python 3.11 from <https://www.python.org/downloads/windows/>. In the installer tick **"Add python.exe to PATH"** and keep the **py launcher** option enabled. Check it in PowerShell:

```powershell
py -3.11 --version
```

### 5.2 Open the project folder in PowerShell
Unzip `FinPilot_GenAI_Finance_Advisor.zip`, then (adjust the path):

```powershell
cd C:\Users\YourName\Downloads\FinPilot
```

### 5.3 Create and activate a virtual environment, then install dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The install downloads PyTorch (needed by Sentence Transformers), so it can take several minutes and a few hundred MB. Your prompt shows `(.venv)` when the environment is active.

> **If PowerShell says "running scripts is disabled"**, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> (or use Command Prompt: `.venv\Scripts\activate.bat`).

### 5.4 Activating the environment in a new terminal
Every time you open a **new** terminal, go to the project folder and activate it again:

```powershell
cd C:\Users\YourName\Downloads\FinPilot
.\.venv\Scripts\Activate.ps1
```

## 6. Get a free Gemini API key (Google AI Studio)

1. Open <https://aistudio.google.com/> and sign in with a Google account.
2. Choose **Get API key** and create a key.
3. In the project folder copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
4. Edit the two lines:
   ```
   GEMINI_API_KEY=paste_your_real_key_here
   GEMINI_MODEL=gemini-3.1-flash-lite
   ```
5. Save the file and **restart** the app.

**Important notes about the free tier**

- Google decides which models are free and what the limits are (requests per minute/day). These change over time, so check the **rate limit** page in Google AI Studio. If you see a "quota" message in FinPilot, wait a minute or choose another model.
- Model names are retired regularly. FinPilot does **not** hardcode a model: change `GEMINI_MODEL` in `.env` whenever needed. If you get a "model not found" message, pick a currently supported model from the AI Studio model list (for example a newer Flash or Flash-Lite model).
- Never share your `.env` file or post your key online. `.env` is excluded from Git and from the ZIP.

You can use FinPilot **without a key**: the dashboard, CSV analysis, document indexing and retrieval inspection still work. Only AI-generated answers are disabled (and FinPilot never fakes them).

## 7. Run the application

```powershell
streamlit run app.py
```

or double-click **`run_windows.bat`**. The app opens at <http://localhost:8501>.

The **first time** you index documents, Sentence Transformers downloads the embedding model (about 90 MB, internet needed once). After that it is cached on your computer.

## 8. Run the tests

```powershell
python -m pytest -v
```

or double-click **`run_tests.bat`**. The main tests run **without** an API key or internet connection. They cover calculations, CSV validation, chunking, retrieval, citations, Gemini error handling (with test doubles) and Streamlit page smoke tests.

Optional integration tests (real embedding model, real Gemini) are skipped by default. To run them:

```powershell
$env:RUN_INTEGRATION=1
python -m pytest -v -m integration
```

### About the evaluation metrics
The Knowledge Base page has a small **Evaluation** tab: a **retrieval hit rate** (does the expected document appear in the top-k results for 8 test questions?), an off-topic-question check, and a **citation presence** check for the latest answer. These are **basic project metrics**. A high hit rate or valid citation markers do **not** prove that every generated answer is factually correct. Always read the cited passages.

## 9. CSV format

| column | meaning | example |
|---|---|---|
| `date` | transaction date | `2026-04-01` |
| `description` | free text | `Monthly salary credit` |
| `category` | spending/income category | `Groceries` |
| `type` | `income` or `expense` | `expense` |
| `amount` | positive number (₹ and commas allowed) | `2450` |

Click **Download sample CSV** on the dashboard to see a complete example.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `py` is not recognised | Reinstall Python and tick the *py launcher* option, or use `python` instead of `py -3.11`. |
| Activation script blocked | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `activate.bat` in Command Prompt. |
| `ModuleNotFoundError` | The virtual environment is not active, or dependencies are missing. Activate `.venv` and run `pip install -r requirements.txt`. |
| "Gemini API key not set" | Create `.env` from `.env.example`, add your key, restart the app. |
| "API key not valid" | Re-copy the key from Google AI Studio (no quotes, no spaces). |
| "Model not found" | Set `GEMINI_MODEL` in `.env` to a currently supported model. |
| "Rate limit or quota reached" | Free-tier limits apply. Wait a minute, reduce requests, or change model. |
| "Could not load the embedding model" | The first run needs internet to download the model once. Check your connection/firewall. |
| A PDF "contains no extractable text" | It is probably a scanned image. FinPilot has no OCR; use a text-based PDF or a TXT file. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| "The saved index could not be loaded" | Use **Clear entire index** on the Knowledge Base page and index again. |
| Slow first answer in the advisor | When the knowledge base is enabled, the embedding model loads once on first use. |

## 11. Privacy and limitations

**Stays on your computer:** uploaded CSV files, documents, embeddings, the FAISS index and chunk metadata. FinPilot has no database, no accounts and no analytics (Streamlit usage statistics are switched off in `.streamlit/config.toml`).

**Sent to Google's Gemini API** (and processed by Google) when you use AI features:
- your question and the conversation so far;
- on the Advisor page: **summary figures only** (totals, category totals, monthly totals, budget comparison). Transaction descriptions and individual transactions are **not** sent;
- on the Knowledge Base page and (optionally) the Advisor: the few retrieved text passages.

Use the **"Inspect the data that will be sent to Gemini"** panel on the Advisor page and the **"Prompt sent to Gemini"** panel on the Knowledge Base page to see the exact text.

Free-tier API terms may allow Google to use submitted content to improve its products. Check Google's current Gemini API terms, and **use the fictional sample data or non-sensitive information** for demonstrations. Do not upload confidential documents to a free-tier key.

FinPilot never asks for banking credentials, never connects to bank accounts, and never makes payments or trades.

**Known limitations**
- Generated answers can be wrong. Figures come from Python, but the explanation text comes from an LLM.
- No OCR for scanned PDFs. Only PDF and TXT are supported.
- Retrieval quality depends on the small embedding model. Tune chunk size, top-k and the similarity threshold on the page.
- The similarity threshold (default 0.25) is a reasonable starting point, not a universal constant. Check the evaluation tab after indexing your own documents.
- Budget comparison uses average monthly spending or one chosen month. Savings projections use straight-line or fixed-rate assumptions.
- Chart hover and labels use Indian digit grouping; axis tick labels use standard grouping (Plotly limitation).
- One user at a time (local, single-user design).

## 12. How to demonstrate the project (5-7 minute plan)

1. **Intro (30 s):** "FinPilot combines accurate Python calculations with Gemini explanations and a RAG knowledge base."
2. **Dashboard (2 min):** click *Load sample dataset*. Show the KPIs, then the donut chart and monthly trend. Open *Set monthly budgets*, lower the Groceries budget, and show the new overspending warning. Download the analysed CSV. Mention validation by uploading a broken CSV (e.g. a missing `amount` column) to show the error message.
3. **Knowledge Base (2 min):** *Index sample documents*. Ask "How many months of expenses should an emergency fund cover?" with **Retrieve passages only** to show similarity scores, then **Ask Gemini** to show the cited answer. Expand the passages and the prompt. Ask an off-topic question ("How do I bake a cake?") to show that FinPilot refuses to invent an answer. Open the **Evaluation** tab and run the retrieval test.
4. **AI Advisor (1.5 min):** ask "How can I save ₹50,000 in six months?". Point out the "Inspect the data that will be sent to Gemini" panel: calculated figures vs hypothetical estimates, and no transaction descriptions.
5. **Close (30 s):** privacy (local index, only summaries/passages sent), limitations (not professional advice, free-tier limits), and future work (OCR, more evaluation, multi-user support).

Tip: do a dry run first so the embedding model is already downloaded, and keep a screenshot of the working app as a backup in case of no internet.

## 13. Development notes

The automated tests were developed and run on Linux with Python 3.12 and the dependency versions listed in `requirements.txt` (tested versions are noted there). The unit tests use a small deterministic *test embedder* so they run offline; the real application uses Sentence Transformers. The Windows batch files, the real MiniLM embedding model and live Gemini calls could not be exercised in that environment, so the optional integration tests are provided for you to run on your machine.
