# PaperIQ — Research Paper Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://paperiq-5yay8appwgeg5ndj2ouvjni.streamlit.app/)

A Streamlit web app for analyzing academic research papers. Supports PDF and DOCX uploads, with features for summarization, domain classification, keyword extraction, sentiment analysis, coherence scoring, and more.

🔗 **Live app:** https://paperiq-5yay8appwgeg5ndj2ouvjni.streamlit.app/

---

## Features

- **Multi-role auth** — Separate login for Teachers and Students, with registration, security-question-based password reset, and a Teacher admin panel
- **Paper upload** — Accepts `.pdf` and `.docx` research papers
- **Section detection** — Automatically identifies Abstract, Introduction, Methodology, Results, and Conclusion
- **Summarization** — Short / Medium / Long summaries per section
- **Domain & sub-domain classification** — Maps papers to domains like AI, ML, IoT, Cybersecurity, etc.
- **Keyword extraction** — TF-IDF + frequency-based top keywords
- **Sentiment analysis** — Per-section sentiment via TextBlob
- **Coherence & composite scoring** — Quantitative quality scores
- **Research aspects checklist** — Detects presence of dataset, baselines, metrics, ethics, reproducibility, etc.
- **Saved papers** — Bookmark papers with scores and abstract summaries
- **Upload history** — Track every analyzed paper with word/page counts and coherence scores
- **User profile dashboard** — Stats, domain pie chart, and recent activity log
- **PDF export** — Download analysis reports as PDF
- **Teacher admin panel** — View all users and login history

---

## Project Structure

```
paperIQ/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── paperiq_users.db     # SQLite database (auto-created on first run)
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sohith170905/paperIQ.git
cd paperIQ
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pymupdf` is imported as `fitz`. Install it with `pip install pymupdf`.
> `python-docx` is imported as `docx`. Install it with `pip install python-docx`.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Demo Credentials

Use these to try the [live app](https://paperiq-5yay8appwgeg5ndj2ouvjni.streamlit.app/) or a local install. They are demo accounts only.

| Role    | ID / Email              | Password     |
|---------|-------------------------|--------------|
| Teacher | teacher@paperiq.com     | teacher123   |
| Student | STU001                  | student123   |
| Student | STU002                  | student456   |

---

## Supported File Types

| Extension | Description          |
|-----------|----------------------|
| `.pdf`    | Research papers (text-based PDFs) |
| `.docx`   | Microsoft Word documents |

---

## Dependencies

| Package        | Purpose                              |
|----------------|--------------------------------------|
| streamlit      | Web UI framework                     |
| pymupdf        | PDF text extraction (`fitz`)         |
| python-docx    | DOCX text extraction                 |
| numpy          | Numerical operations                 |
| pandas         | Data display (tables, dataframes)    |
| textblob       | Sentiment analysis                   |
| plotly         | Interactive charts                   |
| fpdf2          | PDF report export                    |
| scikit-learn   | TF-IDF vectorization & similarity    |

---

## Database

The app uses **SQLite** (`paperiq_users.db`) auto-created on first run. Tables:

- `teachers` — Teacher accounts
- `students` — Student accounts
- `login_history` — All login events
- `upload_history` — Per-user paper upload log
- `saved_papers` — Bookmarked papers
- `paper_topics` — Domain/sub-domain per analysis
- `paper_summaries_store` — Saved section summaries
- `paper_insights_store` — Saved scores and insights

---

## License

MIT
