import streamlit as st
import fitz  # PyMuPDF — pip install pymupdf
import re, heapq, string, hashlib, sqlite3, time
import numpy as np
import pandas as pd
from collections import Counter
import docx as docx_lib
from textblob import TextBlob
import plotly.graph_objects as go
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PaperIQ - Research Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
STOP_WORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'from','as','is','was','are','were','be','been','being','have','has','had',
    'do','does','did','will','would','should','could','may','might','must','can',
    'this','that','these','those','i','you','he','she','it','we','they','what',
    'which','who','when','where','why','how','all','each','every','both','few',
    'more','most','other','some','such','only','own','same','so','than','too',
    'very','just','about','into','through','during','before','after','above',
    'below','up','down','out','off','over','under','again','further','then','once'
])

DOMAIN_KEYWORDS = {
    'Artificial Intelligence': ['ai','artificial intelligence','neural network','deep learning',
                                'machine learning','reinforcement learning','nlp',
                                'natural language processing','computer vision'],
    'Machine Learning':        ['machine learning','classification','regression','clustering',
                                'supervised','unsupervised','feature extraction','prediction',
                                'model','algorithm','training'],
    'Data Science':            ['data science','data analysis','big data','analytics',
                                'visualization','statistics','data mining','predictive analytics'],
    'Internet of Things':      ['iot','internet of things','sensor','embedded','smart device',
                                'wireless','connectivity','edge computing'],
    'Cybersecurity':           ['security','cybersecurity','encryption','cryptography',
                                'vulnerability','threat','firewall','malware','authentication'],
    'Cloud Computing':         ['cloud','aws','azure','virtualization','containerization',
                                'docker','kubernetes'],
    'Blockchain':              ['blockchain','cryptocurrency','bitcoin','ethereum',
                                'distributed ledger','smart contract'],
    'Computer Networks':       ['network','protocol','routing','switching','tcp','ip',
                                'wireless','bandwidth'],
    'Database Systems':        ['database','sql','nosql','mongodb','mysql','query','data storage'],
    'Agriculture':             ['agriculture','crop','farming','soil','cultivation',
                                'harvest','yield','irrigation'],
}

SUB_DOMAIN_KEYWORDS = {
    'Artificial Intelligence': {
        'Computer Vision':     ['image','vision','object detection','segmentation','cnn','convolutional'],
        'NLP':                 ['nlp','text','language model','bert','transformer','sentiment','parsing'],
        'Reinforcement Learning': ['reinforcement','reward','agent','policy','q-learning','markov'],
        'Generative AI':       ['generative','gan','diffusion','stable diffusion','llm','gpt','chatgpt'],
    },
    'Machine Learning': {
        'Supervised Learning': ['classification','regression','svm','random forest','decision tree'],
        'Unsupervised Learning':['clustering','kmeans','pca','dimensionality reduction','autoencoder'],
        'Deep Learning':       ['deep learning','neural network','lstm','rnn','attention','transformer'],
        'Federated Learning':  ['federated','privacy','distributed training','aggregation'],
    },
    'Data Science': {
        'Data Mining':         ['mining','pattern','association','frequent itemset'],
        'Visualization':       ['visualization','dashboard','plot','chart','visual analytics'],
        'Statistical Analysis':['statistics','regression','hypothesis','p-value','correlation'],
    },
    'Cybersecurity': {
        'Network Security':    ['firewall','intrusion','ids','ips','ddos','packet'],
        'Cryptography':        ['encryption','cryptography','hash','rsa','aes','key exchange'],
        'Malware Analysis':    ['malware','ransomware','virus','trojan','reverse engineering'],
    },
    'Internet of Things': {
        'Edge Computing':      ['edge','fog computing','offloading','latency','real-time'],
        'Smart Systems':       ['smart home','wearable','sensor fusion','embedded'],
    },
}

RESEARCH_ASPECTS = {
    'Dataset Description':        ['dataset','collected data','training data','test set','benchmark','corpus'],
    'Experimental Setup':         ['experiment','setup','configuration','environment','hardware','gpu','cpu'],
    'Comparison with Baselines':  ['baseline','compared to','outperform','state-of-the-art','sota'],
    'Evaluation Metrics':         ['accuracy','f1','precision','recall','auc','rmse','bleu','rouge','metric'],
    'Statistical Significance':   ['p-value','statistical','confidence interval','t-test','significance'],
    'Limitations':                ['limitation','constraint','drawback','weakness','future work'],
    'Related Work':               ['related work','prior work','previous','literature review','survey'],
    'Reproducibility':            ['reproducible','code available','github','open source','implementation'],
    'Ethical Considerations':     ['ethical','bias','fairness','privacy','consent','impact'],
    'Real-World Applicability':   ['application','deployment','real-world','industry','practical','production'],
}

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════
DB_PATH = "paperiq_users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.strip().encode()).hexdigest()

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Teachers table with security question fields
    c.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            email             TEXT UNIQUE NOT NULL,
            password          TEXT NOT NULL,
            name              TEXT NOT NULL,
            security_question TEXT DEFAULT '',
            security_answer   TEXT DEFAULT '',
            created           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for col_def in ["security_question TEXT DEFAULT ''", "security_answer TEXT DEFAULT ''"]:
        try:
            c.execute(f"ALTER TABLE teachers ADD COLUMN {col_def}")
        except Exception:
            pass

    # Students table with security question fields
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id        TEXT UNIQUE NOT NULL,
            password          TEXT NOT NULL,
            name              TEXT NOT NULL,
            security_question TEXT DEFAULT '',
            security_answer   TEXT DEFAULT '',
            created           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for col_def in ["security_question TEXT DEFAULT ''", "security_answer TEXT DEFAULT ''"]:
        try:
            c.execute(f"ALTER TABLE students ADD COLUMN {col_def}")
        except Exception:
            pass

    # Seed demo teacher
    c.execute("SELECT * FROM teachers WHERE email=?", ("teacher@paperiq.com",))
    if not c.fetchone():
        c.execute(
            "INSERT INTO teachers (email,password,name,security_question,security_answer) VALUES(?,?,?,?,?)",
            ("teacher@paperiq.com", hash_password("teacher123"), "Dr. Smith", "What is your pet name?", "fluffy")
        )

    # Seed demo students
    for sid, pwd, name in [("STU001", "student123", "John Doe"), ("STU002", "student456", "Jane Smith")]:
        c.execute("SELECT * FROM students WHERE student_id=?", (sid,))
        if not c.fetchone():
            c.execute(
                "INSERT INTO students (student_id,password,name,security_question,security_answer) VALUES(?,?,?,?,?)",
                (sid, hash_password(pwd), name, "What is your pet name?", "buddy")
            )

    # Login history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            user_name  TEXT NOT NULL,
            role       TEXT NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status     TEXT NOT NULL
        )
    """)

    # Upload history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            file_name   TEXT,
            word_count  INTEGER DEFAULT 0,
            page_count  INTEGER DEFAULT 0,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Saved papers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_papers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          TEXT NOT NULL,
            file_name        TEXT,
            abstract_summary TEXT DEFAULT '',
            composite_score  REAL DEFAULT 0,
            saved_time       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature 6: Add coherence_score to upload_history
    try:
        c.execute("ALTER TABLE upload_history ADD COLUMN coherence_score REAL DEFAULT 0")
    except Exception:
        pass

    # Feature 12: Topics table
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_topics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key   TEXT NOT NULL,
            file_name  TEXT,
            domain     TEXT DEFAULT '',
            sub_domain TEXT DEFAULT '',
            saved_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature 13: Summaries table
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_summaries_store (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key       TEXT NOT NULL,
            file_name      TEXT,
            section        TEXT DEFAULT '',
            summary_short  TEXT DEFAULT '',
            summary_medium TEXT DEFAULT '',
            summary_long   TEXT DEFAULT '',
            saved_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature 13: Insights table
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_insights_store (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key         TEXT NOT NULL,
            file_name        TEXT,
            composite_score  REAL DEFAULT 0,
            coherence_score  REAL DEFAULT 0,
            domain           TEXT DEFAULT '',
            sub_domain       TEXT DEFAULT '',
            word_count       INTEGER DEFAULT 0,
            key_insights     TEXT DEFAULT '',
            research_gaps    TEXT DEFAULT '',
            saved_time       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

# ── Teacher DB functions ────────────────────────────────────────────────────
def db_verify_teacher(email, password):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM teachers WHERE email=? AND password=?",
              (email.strip().lower(), hash_password(password)))
    u = c.fetchone(); conn.close()
    return dict(u) if u else None

def db_register_teacher(email, password, name, sq="", sa=""):
    try:
        conn = get_db(); c = conn.cursor()
        c.execute(
            "INSERT INTO teachers(email,password,name,security_question,security_answer) VALUES(?,?,?,?,?)",
            (email.strip().lower(), hash_password(password), name.strip(), sq.strip(), sa.strip().lower())
        )
        conn.commit(); conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Email already exists."

def db_get_teacher_security_question(email):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT security_question FROM teachers WHERE email=?", (email.strip().lower(),))
    r = c.fetchone(); conn.close()
    return r["security_question"] if r else None

def db_verify_teacher_security_answer(email, answer):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT id FROM teachers WHERE email=? AND security_answer=?",
              (email.strip().lower(), answer.strip().lower()))
    r = c.fetchone(); conn.close()
    return bool(r)

def db_update_teacher_password(email, new_password):
    conn = get_db(); c = conn.cursor()
    c.execute("UPDATE teachers SET password=? WHERE email=?",
              (hash_password(new_password), email.strip().lower()))
    conn.commit(); conn.close()

# ── Student DB functions ────────────────────────────────────────────────────
def db_verify_student(student_id, password):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM students WHERE student_id=? AND password=?",
              (student_id.strip().upper(), hash_password(password)))
    u = c.fetchone(); conn.close()
    return dict(u) if u else None

def db_register_student(student_id, password, name, sq="", sa=""):
    try:
        conn = get_db(); c = conn.cursor()
        c.execute(
            "INSERT INTO students(student_id,password,name,security_question,security_answer) VALUES(?,?,?,?,?)",
            (student_id.strip().upper(), hash_password(password), name.strip(), sq.strip(), sa.strip().lower())
        )
        conn.commit(); conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Student ID already exists."

def db_get_student_security_question(student_id):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT security_question FROM students WHERE student_id=?", (student_id.strip().upper(),))
    r = c.fetchone(); conn.close()
    return r["security_question"] if r else None

def db_verify_student_security_answer(student_id, answer):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT id FROM students WHERE student_id=? AND security_answer=?",
              (student_id.strip().upper(), answer.strip().lower()))
    r = c.fetchone(); conn.close()
    return bool(r)

def db_update_student_password(student_id, new_password):
    conn = get_db(); c = conn.cursor()
    c.execute("UPDATE students SET password=? WHERE student_id=?",
              (hash_password(new_password), student_id.strip().upper()))
    conn.commit(); conn.close()

# ── Common DB functions ─────────────────────────────────────────────────────
def db_log_login(user_id, user_name, role, status="Success"):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO login_history(user_id,user_name,role,status) VALUES(?,?,?,?)",
              (user_id, user_name, role, status))
    conn.commit(); conn.close()

def db_get_login_history():
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM login_history ORDER BY login_time DESC")
    rows = [dict(r) for r in c.fetchall()]; conn.close()
    return rows

def db_get_all_teachers():
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT id,name,email,created FROM teachers ORDER BY created DESC")
    rows = [dict(r) for r in c.fetchall()]; conn.close()
    return rows

def db_get_all_students():
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT id,name,student_id,created FROM students ORDER BY created DESC")
    rows = [dict(r) for r in c.fetchall()]; conn.close()
    return rows

def db_log_upload(user_id, file_name, word_count=0, page_count=0, coherence_score=0):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO upload_history(user_id,file_name,word_count,page_count,coherence_score) VALUES(?,?,?,?,?)",
              (user_id, file_name, word_count, page_count, coherence_score))
    conn.commit(); conn.close()

def db_save_topics(user_key, file_name, domain, sub_domain):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO paper_topics(user_key,file_name,domain,sub_domain) VALUES(?,?,?,?)",
              (user_key, file_name, domain, sub_domain))
    conn.commit(); conn.close()

def db_get_topics(user_key):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM paper_topics WHERE user_key=? ORDER BY saved_time DESC", (user_key,))
    rows = [dict(r) for r in c.fetchall()]; conn.close(); return rows

def db_save_insights(user_key, file_name, composite, coherence, domain, sub_domain, wc, key_ins, gaps):
    conn = get_db(); c = conn.cursor()
    c.execute("""INSERT INTO paper_insights_store
                 (user_key,file_name,composite_score,coherence_score,domain,sub_domain,word_count,key_insights,research_gaps)
                 VALUES(?,?,?,?,?,?,?,?,?)""",
              (user_key, file_name, composite, coherence, domain, sub_domain, wc,
               str(key_ins), str(gaps)))
    conn.commit(); conn.close()

def db_save_summaries(user_key, file_name, sec_summs):
    conn = get_db(); c = conn.cursor()
    for sec, summ_dict in sec_summs.items():
        c.execute("""INSERT INTO paper_summaries_store
                     (user_key,file_name,section,summary_short,summary_medium,summary_long)
                     VALUES(?,?,?,?,?,?)""",
                  (user_key, file_name, sec,
                   summ_dict.get('Short',''), summ_dict.get('Medium',''), summ_dict.get('Long','')))
    conn.commit(); conn.close()

def db_get_upload_history(user_id):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM upload_history WHERE user_id=? ORDER BY upload_time DESC", (user_id,))
    rows = [dict(r) for r in c.fetchall()]; conn.close()
    return rows

def db_save_paper(user_id, file_name, abstract_summary, composite_score):
    conn = get_db(); c = conn.cursor()
    c.execute(
        "INSERT INTO saved_papers(user_id,file_name,abstract_summary,composite_score) VALUES(?,?,?,?)",
        (user_id, file_name, abstract_summary[:500], round(composite_score, 2))
    )
    conn.commit(); conn.close()

def db_get_saved_papers(user_id):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT * FROM saved_papers WHERE user_id=? ORDER BY saved_time DESC", (user_id,))
    rows = [dict(r) for r in c.fetchall()]; conn.close()
    return rows

def db_delete_saved_paper(paper_id):
    conn = get_db(); c = conn.cursor()
    c.execute("DELETE FROM saved_papers WHERE id=?", (paper_id,))
    conn.commit(); conn.close()

init_db()

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "authenticated": False,
        "user_role":     None,
        "user_name":     None,
        "user_id":       None,
        "page":          None,
        "auth_role":     None,
        "sidebar_page":  "dashboard",
        "show_admin":    False,
        "analyses":      {},
        "current_file":  None,
        "summary_length":"Medium",
        # forgot-password wizard
        "fp_step":       1,
        "fp_role":       None,
        "fp_id":         "",
        "fp_sq":         "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (original purple-gradient theme, extended)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);}
.header-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:2rem;border-radius:15px;text-align:center;margin-bottom:2rem;
  box-shadow:0 8px 32px rgba(118,75,162,.4);}
.header-title{color:white;font-size:2.2rem;font-weight:bold;letter-spacing:.2rem;}
.header-subtitle{color:rgba(255,255,255,.9);font-size:1rem;text-transform:uppercase;
  letter-spacing:.1rem;margin-top:.5rem;}
.section-header{color:#f093fb;font-size:1.4rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.12rem;margin-top:2rem;margin-bottom:1.5rem;padding-bottom:.6rem;
  border-bottom:2px solid rgba(118,75,162,.3);}
.stats-card{background:linear-gradient(135deg,rgba(102,126,234,.15) 0%,rgba(118,75,162,.15) 100%);
  padding:2rem 1.5rem;border-radius:15px;text-align:center;border:1px solid rgba(118,75,162,.3);}
.stats-number{font-size:3rem;font-weight:bold;margin-bottom:.5rem;}
.stats-label{color:#f093fb;font-weight:600;font-size:1.1rem;text-transform:uppercase;letter-spacing:.05rem;}
.score-card{background:linear-gradient(135deg,rgba(102,126,234,.18) 0%,rgba(118,75,162,.18) 100%);
  padding:1.5rem 1rem;border-radius:15px;text-align:center;
  border:1px solid rgba(118,75,162,.35);margin-bottom:.5rem;}
.score-number{font-size:2.2rem;font-weight:bold;}
.score-label{color:#f093fb;font-weight:600;font-size:.95rem;text-transform:uppercase;letter-spacing:.05rem;}
.domain-badge{display:inline-block;padding:.5rem 1rem;margin:.3rem;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:white;border-radius:20px;font-size:.9rem;font-weight:600;}
.insight-section{background:linear-gradient(135deg,rgba(102,126,234,.1) 0%,rgba(118,75,162,.1) 100%);
  padding:2rem;border-radius:15px;margin-top:2rem;border:1px solid rgba(118,75,162,.3);}
.insight-title{color:#f093fb;font-size:1.8rem;font-weight:600;margin-bottom:1.5rem;
  text-transform:uppercase;letter-spacing:.1rem;}
.insight-item{margin-bottom:1rem;line-height:1.8;}
.insight-label{color:#f093fb;font-weight:600;font-size:1.1rem;}
.insight-value{color:#e0e0e0;font-size:1rem;}
.summary-box{background:linear-gradient(135deg,rgba(102,126,234,.08) 0%,rgba(118,75,162,.08) 100%);
  border-left:4px solid #764ba2;border-radius:10px;padding:1.4rem 1.8rem;margin-bottom:.8rem;
  color:#e0e0e0;font-size:1rem;line-height:1.9;text-align:justify;}
.sentiment-row{background:linear-gradient(135deg,rgba(102,126,234,.1) 0%,rgba(118,75,162,.1) 100%);
  border-radius:10px;padding:1rem 1.5rem;margin-bottom:.6rem;border:1px solid rgba(118,75,162,.25);}
.kw-badge-found{display:inline-block;padding:.3rem .8rem;margin:.2rem;
  background:rgba(80,200,120,.2);border:1px solid rgba(80,200,120,.5);
  color:#50c878;border-radius:12px;font-size:.85rem;font-weight:600;}
.kw-badge-miss{display:inline-block;padding:.3rem .8rem;margin:.2rem;
  background:rgba(255,100,100,.2);border:1px solid rgba(255,100,100,.5);
  color:#ff6464;border-radius:12px;font-size:.85rem;font-weight:600;}
.stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:white;border:none;padding:.9rem 1.8rem;font-size:1rem;
  border-radius:10px;font-weight:600;transition:all .3s ease;
  box-shadow:0 4px 15px rgba(118,75,162,.3);min-height:52px;}
.stButton>button:hover{background:linear-gradient(135deg,#764ba2 0%,#f093fb 100%);
  transform:translateY(-2px);box-shadow:0 6px 20px rgba(118,75,162,.5);}
.stTextInput>div>div>input{background:rgba(26,26,46,.7);
  border:1px solid rgba(118,75,162,.4);border-radius:10px;
  color:#e0e0e0;padding:.9rem;font-size:1rem;}
.stTextInput>div>div>input:focus{border-color:#764ba2;box-shadow:0 0 0 2px rgba(118,75,162,.3);}
h1,h2,h3,h4,h5,h6{color:#f093fb;}
p,div,span,label{color:#e0e0e0;}
.stAlert{background:rgba(102,126,234,.1);border-left:4px solid #667eea;color:#e0e0e0;}
.streamlit-expanderHeader{background:rgba(102,126,234,.12);border-radius:8px;
  color:#f093fb !important;font-weight:500;}
.landing-logo{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:2.5rem 2rem;border-radius:20px;box-shadow:0 10px 40px rgba(118,75,162,.5);
  margin-bottom:2rem;text-align:center;display:flex;flex-direction:column;
  align-items:center;justify-content:center;}
.landing-logo h1{color:white;font-size:3rem;font-weight:bold;letter-spacing:.3rem;
  margin:0;text-align:center;width:100%;}
.landing-logo p{color:rgba(255,255,255,.9);font-size:1rem;margin-top:.5rem;
  letter-spacing:.1rem;text-align:center;width:100%;}
.role-card{background:linear-gradient(135deg,rgba(102,126,234,.15) 0%,rgba(118,75,162,.15) 100%);
  border:2px solid rgba(118,75,162,.3);border-radius:14px;padding:1.5rem;
  text-align:center;cursor:pointer;transition:all .2s;}
.role-card:hover{border-color:#764ba2;}
.role-card-active{background:linear-gradient(135deg,rgba(102,126,234,.3) 0%,rgba(118,75,162,.3) 100%);
  border:2px solid #764ba2;border-radius:14px;padding:1.5rem;text-align:center;}
.role-emoji{font-size:2.5rem;}
.role-label{color:#f093fb;font-weight:700;font-size:1rem;margin-top:.4rem;}
.divider{border:none;border-top:1px solid rgba(118,75,162,.3);margin:1.2rem 0;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION  — PyMuPDF (fitz)
# ═══════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file):
    text = ""
    try:
        raw = file.read() if hasattr(file, "read") else file
        doc = fitz.open(stream=raw, filetype="pdf")
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = [b for b in blocks if b[6] == 0]          # text only
            blocks.sort(key=lambda b: (b[1], b[0]))             # top→bottom
            for b in blocks:
                text += b[4] + "\n"
        text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)      # fix hyphens
        text = re.sub(r'\n{3,}', '\n\n', text)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text

def extract_text_from_docx(file):
    try:
        doc = docx_lib.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return ""

def clean_text(text):
    return re.sub(r'\n+', '\n', text).strip()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════
def identify_sections_regex(text):
    sections = {"Abstract": "", "Introduction": "", "Methodology": "",
                "Results": "", "Conclusion": ""}
    patterns = {
        "Abstract":     r"(?i)(abstract)(.*?)(introduction|methodology|methods)",
        "Introduction": r"(?i)(introduction)(.*?)(methodology|methods|results|literature)",
        "Methodology":  r"(?i)(methodology|methods)(.*?)(results|discussion|conclusion)",
        "Results":      r"(?i)(results|discussion)(.*?)(conclusion|future)",
        "Conclusion":   r"(?i)(conclusion)(.*)",
    }
    for sec, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL)
        if m:
            sections[sec] = text[m.start(2):m.end(2)].strip()
    return sections

def detect_sections_line_by_line(text):
    """Robust header-based section splitter."""
    mandatory  = {k: "" for k in ["Abstract","Introduction","Methodology","Results","Conclusion"]}
    detected   = {k: False for k in mandatory}
    all_secs   = {}

    def map_header(h):
        h = h.lower().strip()
        if any(x in h for x in ['abstract','summary']):                         return 'Abstract'
        if any(x in h for x in ['introduction','background','overview']):        return 'Introduction'
        if any(x in h for x in ['method','methodology','proposed','approach']):  return 'Methodology'
        if any(x in h for x in ['result','experiment','evaluation','discussion']):return 'Results'
        if any(x in h for x in ['conclusion','future work']):                    return 'Conclusion'
        return None

    hre   = r'^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][a-zA-Z0-9 \-\:]+)$'
    lines = text.split('\n')
    cur_h, cur_b = None, []

    for line in lines:
        ls = line.strip()
        if not ls:
            continue
        if len(ls) < 60 and re.match(hre, ls):
            if cur_h and cur_b:
                ct = "\n".join(cur_b).strip()
                if len(ct) > 50:
                    all_secs[cur_h] = ct
                    mk = map_header(cur_h)
                    if mk:
                        mandatory[mk] += "\n\n" + ct
                        detected[mk]   = True
            cur_h = re.sub(r'^[\d\. ]+', '', ls).strip()
            cur_b = []
        else:
            if cur_h:
                cur_b.append(ls)

    if cur_h and cur_b:
        ct = "\n".join(cur_b).strip()
        if len(ct) > 50:
            all_secs[cur_h] = ct
            mk = map_header(cur_h)
            if mk:
                mandatory[mk] += "\n\n" + ct
                detected[mk]   = True

    return mandatory, detected, all_secs

# ═══════════════════════════════════════════════════════════════════════════
# UPGRADED SCORING ENGINE  (TF-IDF coherence + regex reasoning)
# ═══════════════════════════════════════════════════════════════════════════
def _syllable_count(word):
    word  = word.lower()
    count = sum(1 for i, ch in enumerate(word)
                if ch in "aeiouy" and (i == 0 or word[i-1] not in "aeiouy"))
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)

def _tfidf_coherence(sentences):
    """Average cosine similarity between consecutive sentence pairs."""
    if len(sentences) < 2:
        return 100.0
    try:
        vec  = TfidfVectorizer(stop_words='english', max_features=200)
        mat  = vec.fit_transform([s.lower() for s in sentences])
        sims = [float(cosine_similarity(mat[i], mat[i+1])[0][0])
                for i in range(len(sentences) - 1)]
        return round(min(100.0, (sum(sims) / len(sims)) * 100), 2)
    except Exception:
        return 50.0

def _regex_reasoning(text, sentences):
    """Reasoning via indicator words + cause-effect regex patterns."""
    inds = ['because','since','if','then','implies','leads to','causes','therefore',
            'thus','hence','consequently','as a result','in order to','due to',
            'for this reason','evidence','suggests','indicate']
    ind_sents = sum(1 for s in sentences if any(kw in s.lower() for kw in inds))
    patterns  = [r'\b\w+ causes? \w+', r'\b\w+ leads? to \w+',
                 r'\bif .{1,50} then ', r'\bdue to \w+']
    p_bonus   = sum(20 for p in patterns if re.search(p, text.lower()))
    ratio     = ind_sents / max(len(sentences), 1)
    return round(min(85.0, ratio * 60 + min(p_bonus, 40)), 2)

def _language_score(words, sentences, wc, sc):
    if not wc or not sc:
        return 0.0
    vs  = (len(set(w.lower() for w in words)) / wc) * 50
    asl = wc / sc
    if   asl < 15:  sl = (asl / 15) * 25
    elif asl > 25:  sl = max(0.0, 25 - (asl - 25) * 2)
    else:           sl = 25.0
    wl  = min(sum(len(w) for w in words) / wc / 10, 1.0) * 15
    cx  = min(10.0, (text.count(',') + text.count(';')) / sc * 5) if 'text' in dir() else 5.0
    return round(min(100.0, vs + sl + wl + cx), 2)

def _sophistication_score(words, wc):
    if not wc:
        return 0.0
    awl = sum(len(w) for w in words) / wc
    vd  = len(set(w.lower() for w in words)) / wc
    return round(min(100.0, (min(awl / 10, 1.0) * 50) * 0.6 + vd * 50 * 0.4), 2)

def _readability_score(words, wc, sc):
    if not wc or not sc:
        return 50.0
    syl   = sum(_syllable_count(w) for w in words)
    score = 206.835 - 1.015 * (wc / sc) - 84.6 * (syl / wc)
    return round(max(0.0, min(100.0, score)), 2)

def analyze_full_document(text):
    blob       = TextBlob(text)
    sentences  = [str(s) for s in blob.sentences]
    words      = blob.words
    wc, sc     = len(words), len(sentences)
    if wc == 0 or sc == 0:
        return None

    asl  = wc / sc
    awl  = sum(len(w) for w in words) / wc
    sent = float(blob.sentiment.polarity)
    subj = float(blob.sentiment.subjectivity)
    cw   = [w for w in words if len(w) > 6]

    lang  = _language_score(words, sentences, wc, sc)
    # fix: pass text explicitly for punctuation count
    punct = text.count(',') + text.count(';')
    cx    = min(10.0, punct / sc * 5)
    lang  = round(min(100.0, lang + cx - 5.0), 2)   # re-add properly

    coh   = _tfidf_coherence(sentences)
    rea   = _regex_reasoning(text, sentences)
    soph  = _sophistication_score(words, wc)
    read  = _readability_score(words, wc, sc)
    comp  = round(lang * 0.20 + coh * 0.25 + rea * 0.20 + soph * 0.15 + read * 0.20, 2)

    return {
        "scores": {
            "Language":       round(lang, 2),
            "Coherence":      round(coh, 2),
            "Reasoning":      round(rea, 2),
            "Sophistication": round(soph, 2),
            "Readability":    round(read, 2),
            "Composite":      comp,
        },
        "stats": {
            "word_count":         wc,
            "sentence_count":     sc,
            "avg_sentence_len":   round(asl, 2),
            "avg_word_len":       round(awl, 2),
            "vocab_diversity":    round(len(set(words)) / wc, 2),
            "complex_word_ratio": round(len(cw) / wc, 2),
        },
        "sentiment":    round(sent, 2),
        "subjectivity": round(subj, 2),
        "issues":       [s for s in sentences if len(s.split()) > 30],
        "full_text":    text,
    }

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARIZATION — Short / Medium / Long
# ═══════════════════════════════════════════════════════════════════════════
def _heuristic_summary(text, n):
    if not text or len(text) < 50:
        return text or "No content."
    blob  = TextBlob(text)
    sents = blob.sentences
    if len(sents) <= n:
        return text
    wf = {}
    for w in blob.words:
        wl = w.lower()
        if wl not in STOP_WORDS and wl.isalpha():
            wf[wl] = wf.get(wl, 0) + 1
    if not wf:
        return text
    mf = max(wf.values())
    for w in wf:
        wf[w] /= mf
    ss = {}
    for sent in sents:
        for w in sent.words:
            if w.lower() in wf:
                ss[sent] = ss.get(sent, 0) + wf[w.lower()]
    top = heapq.nlargest(n, ss, key=ss.get)
    return " ".join(str(s) for s in top)

def generate_summaries(text):
    return {
        "Short":  _heuristic_summary(text, 2),
        "Medium": _heuristic_summary(text, 4),
        "Long":   _heuristic_summary(text, 7),
    }

# ═══════════════════════════════════════════════════════════════════════════
# KEYWORD MATCHING
# ═══════════════════════════════════════════════════════════════════════════
def match_keywords(text, kw_str):
    if not kw_str.strip():
        return [], []
    clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    keys  = sorted(set(k.strip().lower() for k in kw_str.split(',') if k.strip()))
    return [k for k in keys if k in clean], [k for k in keys if k not in clean]

# ═══════════════════════════════════════════════════════════════════════════
# WORD FREQUENCY & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
def analyze_word_frequency(text):
    if not text:
        return {}
    t = text.lower().translate(str.maketrans('', '', string.punctuation))
    w = [w for w in t.split() if w not in STOP_WORDS and len(w) > 2]
    return dict(Counter(w).most_common(20))

def generate_insights(abstract, word_freq):
    ins = {"main_focus": "", "summary_statement": "", "domains": [],
           "clarity_rating": "", "technical_depth": ""}
    if not abstract:
        return ins
    if word_freq:
        top = list(word_freq.items())[:5]
        ins["main_focus"] = ", ".join(w for w, _ in top)
    wc = len(abstract.split())
    sc = len(TextBlob(abstract).sentences)
    ins["summary_statement"] = (
        f"This paper contains {wc} words across {sc} sentences, "
        f"focusing on {ins['main_focus']}."
    )
    al = abstract.lower()
    ds = {d: sum(1 for kw in kws if kw in al) for d, kws in DOMAIN_KEYWORDS.items()}
    ds = {d: s for d, s in ds.items() if s > 0}
    if ds:
        ins["domains"] = sorted(ds.items(), key=lambda x: x[1], reverse=True)[:3]
    avg = wc / max(sc, 1)
    ins["clarity_rating"]  = "High" if avg < 15 else ("Medium" if avg < 25 else "Low")
    ins["technical_depth"] = "Detailed" if wc > 150 else ("Moderate" if wc > 80 else "Brief")
    return ins

# ═══════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def _detect_research_gaps(text):
    """Feature 9: Detect which research aspects are covered or missing."""
    tl = text.lower()
    gaps = {}
    for aspect, keywords in RESEARCH_ASPECTS.items():
        gaps[aspect] = any(kw in tl for kw in keywords)
    return gaps

def _classify_sub_domain(text, primary_domain):
    """Feature 10: Classify into a sub-domain of the primary domain."""
    tl = text.lower()
    sub_map = SUB_DOMAIN_KEYWORDS.get(primary_domain, {})
    best_sub, best_count = "General", 0
    for sub, kws in sub_map.items():
        count = sum(1 for kw in kws if kw in tl)
        if count > best_count:
            best_count, best_sub = count, sub
    return best_sub

def _generate_highlights(text):
    """Feature 11: Extract key highlight sentences."""
    patterns = [
        r'\bwe propose\b', r'\bwe present\b', r'\bwe introduce\b', r'\bwe show\b',
        r'\bwe demonstrate\b', r'\bwe find\b', r'\bour (approach|method|model|system)\b',
        r'\bnovel\b', r'\bstate[- ]of[- ]the[- ]art\b', r'\bsignificant(ly)?\b',
        r'\btherefore\b', r'\bconclude\b', r'\bconclusion\b', r'\bkey (finding|contribution|result)\b',
        r'\bimprove(ment|s)?\b', r'\boutperform\b', r'\bachieve\b',
    ]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 40]
    highlights = []
    for sent in sentences:
        for pat in patterns:
            if re.search(pat, sent.lower()):
                highlights.append(sent)
                break
    return highlights[:15]

def run_analysis(raw_text, kw_str=""):
    text    = clean_text(raw_text)
    res     = analyze_full_document(text)
    if res is None:
        return None

    mmap, sdet, all_secs = detect_sections_line_by_line(text)
    regex_secs            = identify_sections_regex(text)

    # Merge: prefer line-by-line, fallback regex, fallback first 1000 chars
    merged = {}
    for sec in ["Abstract","Introduction","Methodology","Results","Conclusion"]:
        content = mmap.get(sec,"").strip() or regex_secs.get(sec,"").strip()
        if not content and sec == "Abstract":
            content = text[:1000]
        merged[sec] = content

    sec_summaries = {sec: generate_summaries(cont) for sec, cont in merged.items()}
    abstract      = merged.get("Abstract","") or text[:1000]
    word_freq     = analyze_word_frequency(abstract)
    insights      = generate_insights(abstract, word_freq)
    pkw, mkw      = match_keywords(text, kw_str)
    paper_sums    = generate_summaries(abstract + "\n" + merged.get("Introduction",""))

    # Feature 10: Sub-domain classification
    primary_domain = insights["domains"][0][0] if insights.get("domains") else "General"
    sub_domain = _classify_sub_domain(text, primary_domain)

    # Feature 9: Research gap detection
    research_gaps = _detect_research_gaps(text)

    # Feature 11: Insight highlights
    highlights = _generate_highlights(text)

    return {
        "res":              res,
        "merged_sections":  merged,
        "section_detected": sdet,
        "all_secs":         all_secs,
        "section_summaries":sec_summaries,
        "paper_summaries":  paper_sums,
        "word_freq":        word_freq,
        "insights":         insights,
        "present_kw":       pkw,
        "missing_kw":       mkw,
        "sub_domain":       sub_domain,
        "research_gaps":    research_gaps,
        "highlights":       highlights,
        "primary_domain":   primary_domain,
    }

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def _safe(s):
    return str(s).encode('latin-1', 'replace').decode('latin-1')

def create_single_pdf(fname, res, sec_summs, insights, pkw, mkw):
    scores = res["scores"]
    stats  = res["stats"]
    pdf    = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=_safe(f"PaperIQ Analysis Report: {fname}"), ln=1, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=_safe(f"Final Composite Score: {scores['Composite']}/100"), ln=1)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Scores:", ln=1)
    pdf.set_font("Arial", size=12)
    for k, v in scores.items():
        if k != "Composite":
            pdf.cell(200, 10, txt=_safe(f"  {k}: {v}/100"), ln=1)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Text Statistics:", ln=1)
    pdf.set_font("Arial", size=12)
    for k in ["word_count","sentence_count","avg_sentence_len","avg_word_len",
              "vocab_diversity","complex_word_ratio"]:
        pdf.cell(200, 10, txt=_safe(f"  {k.replace('_',' ').title()}: {stats.get(k,'')}"), ln=1)
    if insights.get("domains"):
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10,
                 txt=_safe("Domains: " + ", ".join(d for d, _ in insights["domains"])), ln=1)
    if pkw:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=_safe(f"Keywords Present: {', '.join(pkw)}"), ln=1)
    if mkw:
        pdf.cell(200, 10, txt=_safe(f"Keywords Missing: {', '.join(mkw)}"), ln=1)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Section Summaries (Medium):", ln=1)
    pdf.set_font("Arial", size=10)
    for sec in ["Abstract","Introduction","Methodology","Results","Conclusion"]:
        summ = sec_summs.get(sec, {}).get("Medium", "N/A")
        pdf.multi_cell(0, 10, txt=_safe(f"[{sec}]\n{summ}\n"))
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10,
             txt=_safe(f"Sentiment: {res['sentiment']}  |  Subjectivity: {res['subjectivity']}"), ln=1)
    return pdf.output(dest='S').encode('latin-1')

def create_combined_pdf(all_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for fname, data in all_data.items():
        res    = data["res"]
        scores = res["scores"]
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=_safe(f"PaperIQ Report: {fname}"), ln=1, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=_safe(f"Composite Score: {scores['Composite']}/100"), ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Scores:", ln=1)
        pdf.set_font("Arial", size=12)
        for k, v in scores.items():
            if k != "Composite":
                pdf.cell(200, 10, txt=_safe(f"  {k}: {v}/100"), ln=1)
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Section Summaries (Medium):", ln=1)
        pdf.set_font("Arial", size=10)
        for sec in ["Abstract","Introduction","Methodology","Results","Conclusion"]:
            summ = data["section_summaries"].get(sec, {}).get("Medium", "N/A")
            pdf.multi_cell(0, 10, txt=_safe(f"[{sec}]\n{summ}\n"))
    return pdf.output(dest='S').encode('latin-1')

def generate_markdown(fname, res, sec_summs, insights, pkw, mkw):
    scores = res["scores"]
    stats  = res["stats"]
    md  = f"# PaperIQ Analysis Report: {fname}\n\n"
    md += f"## Final Composite Score: {scores['Composite']}/100\n\n"
    md += "### Scores\n"
    for k, v in scores.items():
        if k != "Composite":
            md += f"- **{k}**: {v}/100\n"
    md += "\n### Text Statistics\n"
    for k in ["word_count","sentence_count","avg_sentence_len","avg_word_len",
              "vocab_diversity","complex_word_ratio"]:
        md += f"- **{k.replace('_',' ').title()}**: {stats.get(k,'')}\n"
    if insights.get("domains"):
        md += "\n### Identified Domains\n"
        md += ", ".join(f"{d} ({s})" for d, s in insights["domains"]) + "\n"
    if pkw or mkw:
        md += "\n### Keywords\n"
        if pkw: md += f"- **Present**: {', '.join(pkw)}\n"
        if mkw: md += f"- **Missing**: {', '.join(mkw)}\n"
    md += "\n### Section Summaries (Medium)\n"
    for sec in ["Abstract","Introduction","Methodology","Results","Conclusion"]:
        summ = sec_summs.get(sec, {}).get("Medium", "N/A")
        md += f"**{sec}**\n\n{summ}\n\n"
    md += f"### Sentiment\n"
    md += f"- Polarity: {res['sentiment']}\n"
    md += f"- Subjectivity: {res['subjectivity']}\n"
    return md

def generate_csv(fname, res, sec_summs, insights, pkw, mkw):
    """Feature 8: CSV export."""
    import io, csv
    scores = res["scores"]
    stats  = res["stats"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Field", "Value"])
    w.writerow(["File", fname])
    w.writerow(["Composite Score", scores.get("Composite",0)])
    w.writerow(["Language Score",  scores.get("Language",0)])
    w.writerow(["Coherence Score", scores.get("Coherence",0)])
    w.writerow(["Reasoning Score", scores.get("Reasoning",0)])
    w.writerow(["Word Count",      stats.get("word_count",0)])
    w.writerow(["Sentence Count",  stats.get("sentence_count",0)])
    w.writerow(["Vocab Diversity", stats.get("vocab_diversity",0)])
    w.writerow(["Sentiment",       res.get("sentiment",0)])
    w.writerow(["Subjectivity",    res.get("subjectivity",0)])
    w.writerow(["Main Focus",      insights.get("main_focus","")])
    w.writerow(["Keywords Present",", ".join(pkw)])
    w.writerow(["Keywords Missing",", ".join(mkw)])
    for sec in ["Abstract","Introduction","Methodology","Results","Conclusion"]:
        w.writerow([f"{sec} Summary", sec_summs.get(sec,{}).get("Medium","N/A")])
    return buf.getvalue().encode("utf-8")

# ═══════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ═══════════════════════════════════════════════════════════════════════════
def show_landing():
    """Feature 1: Enhanced landing page with feature cards."""
    st.markdown("""
    <style>
    .feature-grid{display:flex;gap:1.5rem;margin:2rem 0;flex-wrap:wrap;justify-content:center;}
    .feature-card{background:linear-gradient(135deg,rgba(102,126,234,.18) 0%,rgba(118,75,162,.18) 100%);
      border:1px solid rgba(118,75,162,.4);border-radius:18px;padding:2rem 1.5rem;
      text-align:center;flex:1;min-width:200px;max-width:260px;transition:transform .2s;}
    .feature-card:hover{transform:translateY(-4px);}
    .feature-icon{font-size:2.5rem;margin-bottom:.8rem;}
    .feature-title{color:#f093fb;font-weight:700;font-size:1.1rem;margin-bottom:.5rem;}
    .feature-desc{color:#b0b0c8;font-size:.9rem;line-height:1.5;}
    .landing-tagline{color:#e0e0e0;text-align:center;font-size:1.2rem;
      margin:1.5rem 0;letter-spacing:.04rem;font-style:italic;}
    .stat-card{background:linear-gradient(135deg,rgba(102,126,234,.12) 0%,rgba(118,75,162,.12) 100%);
      border:1px solid rgba(118,75,162,.3);border-radius:14px;padding:1.5rem 1rem;text-align:center;}
    .stat-val{font-size:2rem;font-weight:700;color:#f093fb;}
    .stat-lbl{color:#b0b0c8;font-size:.85rem;text-transform:uppercase;letter-spacing:.05rem;margin-top:.3rem;}
    .activity-item{background:rgba(102,126,234,.08);border-radius:10px;padding:.8rem 1.2rem;
      margin-bottom:.5rem;border-left:3px solid #764ba2;}
    .activity-time{color:#888;font-size:.8rem;float:right;}
    .preview-box{background:rgba(26,26,46,.7);border:1px solid rgba(118,75,162,.3);
      border-radius:10px;padding:1rem 1.5rem;color:#c0c0d8;font-size:.9rem;
      max-height:200px;overflow-y:auto;font-family:monospace;line-height:1.6;}
    .gap-present{display:inline-block;padding:.3rem .8rem;margin:.2rem;
      background:rgba(80,200,120,.15);border:1px solid rgba(80,200,120,.4);
      color:#50c878;border-radius:10px;font-size:.85rem;}
    .gap-missing{display:inline-block;padding:.3rem .8rem;margin:.2rem;
      background:rgba(255,100,100,.12);border:1px solid rgba(255,100,100,.3);
      color:#ff8080;border-radius:10px;font-size:.85rem;}
    .highlight-sentence{background:rgba(240,147,251,.12);border-left:3px solid #f093fb;
      border-radius:8px;padding:.8rem 1.2rem;margin-bottom:.7rem;color:#e0e0e0;
      font-size:.95rem;line-height:1.7;}
    .subdomain-badge{display:inline-block;padding:.4rem 1rem;margin:.2rem;
      background:linear-gradient(135deg,rgba(240,147,251,.2),rgba(118,75,162,.2));
      border:1px solid rgba(240,147,251,.4);color:#f093fb;border-radius:15px;
      font-size:.85rem;font-weight:600;}
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="landing-logo"><h1>PAPERIQ</h1>'
                    '<p>AI-POWERED RESEARCH ANALYZER</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="landing-tagline">✨ Elevate Your Academic Writing with AI-Powered Insights</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Precision Analysis</div>
            <div class="feature-desc">Deep scoring across Language, Coherence, Reasoning &amp; more with detailed breakdowns.</div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Metrics</div>
            <div class="feature-desc">Interactive radar charts, word clouds, sentiment gauges and score visualizations.</div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Instant Feedback</div>
            <div class="feature-desc">Research gap detection, vocabulary suggestions and AI-generated summaries in seconds.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀  Get Started Now", use_container_width=True, key="go_signin"):
                st.session_state.page = "signin"
                st.session_state.auth_role = None
                st.rerun()
        with c2:
            if st.button("📝  Create Account", use_container_width=True, key="go_signup"):
                st.session_state.page = "signup"
                st.session_state.auth_role = None
                st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#888;font-size:.9rem;'>"
                    "New here? Click <b>Create Account</b>.<br>"
                    "Already have one? Click <b>Get Started Now</b>.</p>", unsafe_allow_html=True)


def show_forgot_password():
    """3-step forgot-password flow: role → security-Q → new password."""
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="landing-logo"><h1>PAPERIQ</h1>'
                    '<p>RESET PASSWORD</p></div>', unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;color:#f093fb;'>Forgot Password</h4>",
                    unsafe_allow_html=True)

        role = st.session_state.fp_role
        step = st.session_state.fp_step

        # ── Step 0: select role ─────────────────────────────────────────────
        if not role:
            st.markdown("<p style='text-align:center;color:#aaa;'>"
                        "Select your role to continue</p>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                if st.button("🎓  Teacher", use_container_width=True, key="fp_sel_t"):
                    st.session_state.fp_role = "teacher"
                    st.rerun()
            with r2:
                if st.button("📚  Student", use_container_width=True, key="fp_sel_s"):
                    st.session_state.fp_role = "student"
                    st.rerun()

        # ── Step 1: enter identifier ────────────────────────────────────────
        elif step == 1:
            st.markdown(f"<p style='text-align:center;color:#aaa;'>Role: "
                        f"<b style='color:#f093fb;'>{role.capitalize()}</b></p>",
                        unsafe_allow_html=True)
            if role == "teacher":
                ident = st.text_input("Registered Email",
                                      placeholder="teacher@school.com", key="fp_ident")
                if st.button("Next →", use_container_width=True, key="fp_next1"):
                    sq = db_get_teacher_security_question(ident)
                    if sq:
                        st.session_state.fp_id   = ident
                        st.session_state.fp_sq   = sq
                        st.session_state.fp_step = 2
                        st.rerun()
                    else:
                        st.error("No account found with this email.")
            else:
                ident = st.text_input("Student ID", placeholder="e.g. STU001", key="fp_ident")
                if st.button("Next →", use_container_width=True, key="fp_next1"):
                    sq = db_get_student_security_question(ident)
                    if sq:
                        st.session_state.fp_id   = ident
                        st.session_state.fp_sq   = sq
                        st.session_state.fp_step = 2
                        st.rerun()
                    else:
                        st.error("No account found with this ID.")

        # ── Step 2: answer security question ───────────────────────────────
        elif step == 2:
            st.info(f"🔐 Security Question: **{st.session_state.fp_sq}**")
            answer = st.text_input("Your Answer", key="fp_answer")
            if st.button("Verify Answer", use_container_width=True, key="fp_verify"):
                if role == "teacher":
                    ok = db_verify_teacher_security_answer(st.session_state.fp_id, answer)
                else:
                    ok = db_verify_student_security_answer(st.session_state.fp_id, answer)
                if ok:
                    st.session_state.fp_step = 3
                    st.rerun()
                else:
                    st.error("Incorrect answer. Please try again.")

        # ── Step 3: set new password ────────────────────────────────────────
        elif step == 3:
            st.success("✅ Identity verified! Set your new password.")
            np1 = st.text_input("New Password",     type="password", key="fp_np1")
            np2 = st.text_input("Confirm Password", type="password", key="fp_np2")
            if st.button("Save New Password", use_container_width=True, key="fp_save"):
                if not np1 or not np2:
                    st.error("Fields cannot be empty.")
                elif np1 != np2:
                    st.error("Passwords do not match.")
                elif len(np1) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    if role == "teacher":
                        db_update_teacher_password(st.session_state.fp_id, np1)
                    else:
                        db_update_student_password(st.session_state.fp_id, np1)
                    st.success("Password updated! Redirecting to Sign In...")
                    time.sleep(1.5)
                    st.session_state.page    = "signin"
                    st.session_state.fp_step = 1
                    st.session_state.fp_role = None
                    st.session_state.fp_id   = ""
                    st.session_state.fp_sq   = ""
                    st.rerun()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="fp_back"):
            st.session_state.page    = "signin"
            st.session_state.fp_step = 1
            st.session_state.fp_role = None
            st.session_state.fp_id   = ""
            st.session_state.fp_sq   = ""
            st.rerun()


def show_role_selector(page_type):
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="landing-logo"><h1>PAPERIQ</h1>'
                    '<p>AI-POWERED RESEARCH ANALYZER</p></div>', unsafe_allow_html=True)
        label = "Sign In" if page_type == "signin" else "Sign Up"
        st.markdown(f"<h4 style='text-align:center;color:#f093fb;margin-bottom:.5rem;'>"
                    f"{label}</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#aaa;font-size:.9rem;"
                    "margin-bottom:1.5rem;'>Select your role to continue</p>",
                    unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            ts = "role-card-active" if st.session_state.auth_role == "teacher" else "role-card"
            st.markdown(f'<div class="{ts}"><div class="role-emoji">🎓</div>'
                        f'<div class="role-label">TEACHER</div></div>', unsafe_allow_html=True)
            if st.button("Select Teacher", use_container_width=True, key=f"{page_type}_sel_t"):
                st.session_state.auth_role = "teacher"
                st.rerun()
        with r2:
            ss = "role-card-active" if st.session_state.auth_role == "student" else "role-card"
            st.markdown(f'<div class="{ss}"><div class="role-emoji">📚</div>'
                        f'<div class="role-label">STUDENT</div></div>', unsafe_allow_html=True)
            if st.button("Select Student", use_container_width=True, key=f"{page_type}_sel_s"):
                st.session_state.auth_role = "student"
                st.rerun()

        if st.session_state.auth_role:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            if page_type == "signin":
                _render_signin_form(st.session_state.auth_role)
            else:
                _render_signup_form(st.session_state.auth_role)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        other_label = "Sign Up"  if page_type == "signin" else "Sign In"
        other_page  = "signup"   if page_type == "signin" else "signin"
        note = "Don't have an account?" if page_type == "signin" else "Already have an account?"
        st.markdown(f"<p style='text-align:center;color:#aaa;font-size:.9rem;'>{note}</p>",
                    unsafe_allow_html=True)
        cb, cs = st.columns(2)
        with cb:
            if st.button("← Back", use_container_width=True, key=f"{page_type}_back"):
                st.session_state.page = None
                st.session_state.auth_role = None
                st.rerun()
        with cs:
            if st.button(other_label, use_container_width=True, key=f"switch_{other_page}"):
                st.session_state.page = other_page
                st.session_state.auth_role = None
                st.rerun()


def _login_as(user_id, user_name, role):
    """Fully wipe session state then set new user — prevents data leaking between accounts."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    # Re-init all defaults cleanly
    st.session_state.authenticated = True
    st.session_state.user_role     = role
    st.session_state.user_name     = user_name
    st.session_state.user_id       = user_id
    st.session_state.page          = None
    st.session_state.auth_role     = None
    st.session_state.sidebar_page  = "dashboard"
    st.session_state.show_admin    = False
    st.session_state.analyses      = {}
    st.session_state.current_file  = None
    st.session_state.summary_length= "Medium"
    st.session_state.fp_step       = 1
    st.session_state.fp_role       = None
    st.session_state.fp_id         = ""
    st.session_state.fp_sq         = ""

def _render_signin_form(role):
    if role == "teacher":
        email    = st.text_input("Email",    placeholder="teacher@paperiq.com", key="si_t_email")
        password = st.text_input("Password", type="password",                   key="si_t_pass")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True, key="si_t_btn"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = db_verify_teacher(email, password)
                    if user:
                        db_log_login(user["email"], user["name"], "Teacher", "Success")
                        _login_as(user["email"], user["name"], "teacher")
                        st.rerun()
                    else:
                        db_log_login(email, "Unknown", "Teacher", "Failed")
                        st.error("Incorrect email or password.")
        with c2:
            if st.button("Forgot Password?", use_container_width=True, key="fp_t_link"):
                st.session_state.page    = "forgot_password"
                st.session_state.fp_role = "teacher"
                st.session_state.fp_step = 1
                st.rerun()
        st.markdown("<p style='text-align:center;color:#888;font-size:.8rem;margin-top:.5rem;'>"
                    "Demo: teacher@paperiq.com / teacher123</p>", unsafe_allow_html=True)
    else:
        sid      = st.text_input("Student ID", placeholder="e.g. STU001", key="si_s_id")
        password = st.text_input("Password",   type="password",            key="si_s_pass")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True, key="si_s_btn"):
                if not sid or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = db_verify_student(sid, password)
                    if user:
                        db_log_login(user["student_id"], user["name"], "Student", "Success")
                        _login_as(user["student_id"], user["name"], "student")
                        st.rerun()
                    else:
                        db_log_login(sid, "Unknown", "Student", "Failed")
                        st.error("Incorrect Student ID or password.")
        with c2:
            if st.button("Forgot Password?", use_container_width=True, key="fp_s_link"):
                st.session_state.page    = "forgot_password"
                st.session_state.fp_role = "student"
                st.session_state.fp_step = 1
                st.rerun()
        st.markdown("<p style='text-align:center;color:#888;font-size:.8rem;margin-top:.5rem;'>"
                    "Demo: STU001 / student123</p>", unsafe_allow_html=True)


def _render_signup_form(role):
    if role == "teacher":
        name    = st.text_input("Full Name",        placeholder="Dr. John Smith",       key="su_t_name")
        email   = st.text_input("Email",            placeholder="yourname@school.com",  key="su_t_email")
        pw      = st.text_input("Password",         type="password",                    key="su_t_pass")
        conf    = st.text_input("Confirm Password", type="password",                    key="su_t_conf")
        sq      = st.text_input("Security Question",placeholder="e.g. Your pet name?", key="su_t_sq")
        sa      = st.text_input("Security Answer",  placeholder="Answer (lowercase)",   key="su_t_sa")
        if st.button("Create Account", use_container_width=True, key="su_t_btn"):
            if not all([name, email, pw, conf, sq, sa]):
                st.error("All fields are required.")
            elif pw != conf:    st.error("Passwords do not match.")
            elif len(pw) < 6:   st.error("Password must be at least 6 characters.")
            else:
                ok, msg = db_register_teacher(email, pw, name, sq, sa)
                if ok:
                    st.success("Account created! Please Sign In.")
                    st.session_state.page = "signin"
                    st.session_state.auth_role = "teacher"
                    st.rerun()
                else:
                    st.error(msg)
    else:
        name    = st.text_input("Full Name",        placeholder="Your Full Name",        key="su_s_name")
        sid     = st.text_input("Student ID",       placeholder="e.g. STU003",           key="su_s_id")
        pw      = st.text_input("Password",         type="password",                     key="su_s_pass")
        conf    = st.text_input("Confirm Password", type="password",                     key="su_s_conf")
        sq      = st.text_input("Security Question",placeholder="e.g. Your pet name?",  key="su_s_sq")
        sa      = st.text_input("Security Answer",  placeholder="Answer (lowercase)",    key="su_s_sa")
        if st.button("Create Account", use_container_width=True, key="su_s_btn"):
            if not all([name, sid, pw, conf, sq, sa]):
                st.error("All fields are required.")
            elif pw != conf:    st.error("Passwords do not match.")
            elif len(pw) < 6:   st.error("Password must be at least 6 characters.")
            else:
                ok, msg = db_register_student(sid, pw, name, sq, sa)
                if ok:
                    st.success("Account created! Please Sign In.")
                    st.session_state.page = "signin"
                    st.session_state.auth_role = "student"
                    st.rerun()
                else:
                    st.error(msg)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def show_sidebar():
    with st.sidebar:
        icon = "🎓" if st.session_state.user_role == "teacher" else "📚"
        st.markdown(f"""
        <div style='text-align:center;padding:1rem 0;'>
            <div style='font-size:2.5rem;'>{icon}</div>
            <div style='color:#f093fb;font-weight:700;font-size:1rem;margin-top:.3rem;'>
                {st.session_state.user_name}</div>
            <div style='color:#888;font-size:.8rem;'>
                {st.session_state.user_role.capitalize()}</div>
            <div style='color:#666;font-size:.75rem;margin-top:.2rem;'>
                ID: {st.session_state.user_id}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        nav_items = [
            ("📊  Dashboard",      "dashboard"),
            ("💾  Saved Papers",   "saved"),
            ("📋  Upload History", "history"),
            ("👤  Profile",        "profile"),
        ]
        for label, key in nav_items:
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.sidebar_page = key
                st.rerun()

        st.markdown("---")
        if st.session_state.user_role == "teacher":
            if st.button("🔧  Admin Panel", use_container_width=True, key="admin_btn"):
                st.session_state.show_admin = not st.session_state.get("show_admin", False)
                st.rerun()
            st.markdown("---")

        if st.button("🚪  Logout", use_container_width=True, key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session_state()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════
def show_admin_panel():
    st.markdown('<div class="section-header">ADMIN PANEL</div>', unsafe_allow_html=True)
    atab1, atab2, atab3 = st.tabs(["📊 Login History", "🎓 All Teachers", "📚 All Students"])

    with atab1:
        st.markdown("#### Login History (All Users)")
        history = db_get_login_history()
        if history:
            total    = len(history)
            success  = sum(1 for h in history if h["status"] == "Success")
            failed   = total - success
            teachers = sum(1 for h in history if h["role"] == "Teacher" and h["status"] == "Success")
            students = sum(1 for h in history if h["role"] == "Student" and h["status"] == "Success")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Logins",     total)
            m2.metric("Successful",       success)
            m3.metric("Failed",           failed)
            m4.metric("Teachers/Students", f"{teachers}/{students}")
            df = pd.DataFrame(history).rename(columns={
                "id": "Sr.", "user_id": "User ID", "user_name": "Name",
                "role": "Role", "login_time": "Login Time", "status": "Status"
            })
            st.dataframe(df[["Sr.","User ID","Name","Role","Login Time","Status"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No login history yet.")

    with atab2:
        tlist = db_get_all_teachers()
        if tlist:
            df_t = pd.DataFrame(tlist).rename(columns={
                "id":"ID","name":"Name","email":"Email","created":"Registered On"})
            st.dataframe(df_t, use_container_width=True, hide_index=True)
            st.success(f"Total Teachers: {len(tlist)}")
        else:
            st.info("No teachers registered yet.")

    with atab3:
        slist = db_get_all_students()
        if slist:
            df_s = pd.DataFrame(slist).rename(columns={
                "id":"ID","name":"Name","student_id":"Student ID","created":"Registered On"})
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            st.success(f"Total Students: {len(slist)}")
        else:
            st.info("No students registered yet.")
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD — multi-file upload, analysis, all 7 tabs
# ═══════════════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown('<div class="section-header">UPLOAD RESEARCH PAPERS</div>',
                unsafe_allow_html=True)

    col_up, col_manual = st.columns([1.2, 1])
    with col_up:
        st.markdown("#### Option A: Upload Files (PDF / DOCX / TXT)")
        uploaded_files = st.file_uploader(
            "Drop one or more files here",
            type=["pdf","docx","txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
    with col_manual:
        st.markdown("#### Option B: Paste Text Directly")
        manual_title = st.text_input("Paper Title (optional)", key="manual_title")
        manual_text  = st.text_area("Paste abstract or full text", height=110, key="manual_text")

    # Controls row
    sc1, sc2 = st.columns(2)
    with sc1:
        sum_len = st.select_slider(
            "📏  Summary Detail Level",
            options=["Short","Medium","Long"],
            value=st.session_state.summary_length,
            key="sum_slider"
        )
        st.session_state.summary_length = sum_len
    with sc2:
        kw_str = st.text_input(
            "🔑  Keywords to check (comma-separated)",
            placeholder="e.g. machine learning, neural network",
            key="kw_input"
        )

    # Feature 4: Document Preview
    show_preview = st.checkbox("👁️  Preview extracted text before analysis", key="preview_toggle")
    if show_preview and uploaded_files:
        st.markdown("#### Document Preview")
        for uf in uploaded_files[:3]:
            try:
                if uf.name.endswith(".pdf"):
                    preview_text = extract_text_from_pdf(uf)
                    uf.seek(0)
                elif uf.name.endswith(".docx"):
                    preview_text = extract_text_from_docx(uf)
                    uf.seek(0)
                else:
                    preview_text = uf.getvalue().decode("utf-8")
                    uf.seek(0)
                with st.expander(f"📄 {uf.name}", expanded=True):
                    st.markdown(f'<div class="preview-box">{preview_text[:1500]}…</div>',
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not preview {uf.name}: {e}")

    if st.button("🔍  Analyze Paper(s)", use_container_width=True, key="analyze_btn"):
        st.session_state.analyses    = {}
        st.session_state.current_file = None

        tasks = []
        if manual_text.strip():
            tasks.append((manual_title.strip() or "Manual Text", None, manual_text))
        for uf in (uploaded_files or []):
            tasks.append((uf.name, uf, None))

        if not tasks:
            st.error("Please upload at least one file or paste some text.")
        else:
            prog = st.progress(0)
            for i, (fname, uf, raw) in enumerate(tasks):
                with st.spinner(f"Analyzing {fname}…"):
                    try:
                        if raw:
                            text = raw
                        elif fname.endswith(".pdf"):
                            text = extract_text_from_pdf(uf)
                        elif fname.endswith(".docx"):
                            text = extract_text_from_docx(uf)
                        else:
                            text = uf.getvalue().decode("utf-8")

                        data = run_analysis(text, kw_str)
                        if data:
                            st.session_state.analyses[fname] = data
                            if not st.session_state.current_file:
                                st.session_state.current_file = fname
                            coherence = data["res"]["scores"].get("Coherence", 0)
                            db_log_upload(st.session_state.user_id, fname,
                                          data["res"]["stats"]["word_count"], 0, coherence)
                            # Feature 12: save topics
                            db_save_topics(st.session_state.user_id, fname,
                                           data.get("primary_domain",""), data.get("sub_domain",""))
                            # Feature 13: save insights + summaries
                            try:
                                db_save_insights(st.session_state.user_id, fname,
                                                 data["res"]["scores"].get("Composite",0), coherence,
                                                 data.get("primary_domain",""), data.get("sub_domain",""),
                                                 data["res"]["stats"]["word_count"],
                                                 data["insights"], data.get("research_gaps",{}))
                                db_save_summaries(st.session_state.user_id, fname,
                                                  data["section_summaries"])
                            except Exception:
                                pass
                        else:
                            st.warning(f"Could not extract enough text from '{fname}'.")
                    except Exception as e:
                        st.error(f"Error analyzing '{fname}': {e}")
                prog.progress((i + 1) / len(tasks))

    if not st.session_state.analyses:
        return

    # Feature 5: Analyze Another button
    if st.button("🔄  Analyze Another Paper", key="analyze_another_btn"):
        st.session_state.analyses = {}
        st.session_state.current_file = None
        st.rerun()

    # ── Paper selector ────────────────────────────────────────────────────
    names = list(st.session_state.analyses.keys())
    if len(names) > 1:
        cur = st.selectbox(
            "📂  Select paper to view",
            names,
            index=(names.index(st.session_state.current_file)
                   if st.session_state.current_file in names else 0),
            key="paper_sel"
        )
        st.session_state.current_file = cur
    else:
        st.session_state.current_file = names[0]

    fname        = st.session_state.current_file
    data         = st.session_state.analyses[fname]
    res          = data["res"]
    scores       = res["scores"]
    stats        = res["stats"]
    sec_summs    = data["section_summaries"]
    insights     = data["insights"]
    pkw          = data["present_kw"]
    mkw          = data["missing_kw"]
    merged_secs  = data["merged_sections"]
    sec_det      = data["section_detected"]
    sum_len      = st.session_state.summary_length
    full_text    = res.get("full_text","")
    sub_domain   = data.get("sub_domain", "General")
    research_gaps = data.get("research_gaps", {})
    highlights   = data.get("highlights", [])

    # Paper title
    title_m = re.search(r'^(.{10,200}?)[\.\n]', full_text)
    title   = title_m.group(1) if title_m else fname
    st.markdown(f'<div style="font-size:1.2rem;color:#f093fb;margin:2rem 0 1rem 0;">'
                f'<strong>Paper:</strong> {title}</div>', unsafe_allow_html=True)

    # TL;DR
    tldr = data["paper_summaries"].get(sum_len,"")
    if tldr:
        st.info(f"**✨ AI Quick Summary ({sum_len}):** {tldr}")

    # Metrics row
    st.markdown("### Analysis Results")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Composite Score", f"{scores['Composite']}/100")
    c2.metric("Language",        f"{scores['Language']}/100")
    c3.metric("Coherence",       f"{scores['Coherence']}/100")
    c4.metric("Reasoning",       f"{scores['Reasoning']}/100")

    # Keyword results
    if pkw or mkw:
        kc1, kc2 = st.columns(2)
        if pkw:
            kc1.success(f"✅ Keywords found: {', '.join(pkw)}")
        if mkw:
            kc2.warning(f"⚠️ Keywords missing: {', '.join(mkw)}")

    # Export buttons (Feature 8: added CSV)
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        try:
            pdf_bytes = create_single_pdf(fname, res, sec_summs, insights, pkw, mkw)
            st.download_button("📄 PDF Report", data=pdf_bytes,
                               file_name=f"{fname}_report.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF failed: {e}")
    with ec2:
        md_str = generate_markdown(fname, res, sec_summs, insights, pkw, mkw)
        st.download_button("📝 Markdown", data=md_str,
                           file_name=f"{fname}_report.md", mime="text/markdown")
    with ec3:
        csv_bytes = generate_csv(fname, res, sec_summs, insights, pkw, mkw)
        st.download_button("📊 CSV Export", data=csv_bytes,
                           file_name=f"{fname}_report.csv", mime="text/csv")
    with ec4:
        if len(st.session_state.analyses) > 1:
            try:
                combined = create_combined_pdf(st.session_state.analyses)
                st.download_button("📚 All PDF", data=combined,
                                   file_name="all_papers_report.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Combined PDF failed: {e}")

    # Key Insights card
    st.markdown('<div class="insight-section"><div class="insight-title">Key Insights</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="insight-item"><span class="insight-label">Main Focus: </span>'
                f'<span class="insight-value">{insights["main_focus"]}</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="insight-item"><span class="insight-label">Technical Depth: </span>'
                f'<span class="insight-value">{insights["technical_depth"]}</span></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="insight-item"><span class="insight-value">'
                f'{insights["summary_statement"]}</span></div>', unsafe_allow_html=True)
    if insights.get("domains"):
        dhtml = ('<div style="margin-top:.5rem;">' +
                 "".join(f'<span class="domain-badge">{d} ({s})</span>'
                         for d, s in insights["domains"]) + '</div>')
        st.markdown(f'<div class="insight-item"><span class="insight-label">Domains: </span>'
                    f'{dhtml}</div>', unsafe_allow_html=True)
    # Feature 10: Sub-domain badge
    if sub_domain and sub_domain != "General":
        st.markdown(f'<div class="insight-item"><span class="insight-label">Sub-Domain: </span>'
                    f'<span class="subdomain-badge">🔬 {sub_domain}</span></div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ═══════ TABS ═══════════════════════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
        "📊 Visualizations","📖 Section Summaries","⚠️ Issues",
        "💡 Suggestions","📈 Detailed Metrics","😊 Sentiment","🔍 Cross-Doc Q&A",
        "🧩 Research Gaps","✨ Highlights"
    ])

    # ── Tab 1: Visualizations ────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("#### Metric Radar")
            cats  = ['Language','Coherence','Reasoning','Sophistication','Readability']
            vals  = [scores[k] for k in cats]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself',
                line_color='#764ba2', fillcolor='rgba(118,75,162,0.3)', name='Score'))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100]),
                           bgcolor='rgba(26,26,46,0.8)'),
                paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                height=380, font=dict(color='#e0e0e0'))
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            st.markdown("#### Text Statistics")
            fig_b = go.Figure(data=[
                go.Bar(name='Avg Sentence Length', x=['Sentence Length'],
                       y=[stats['avg_sentence_len']], marker_color='#667eea'),
                go.Bar(name='Avg Word Length',     x=['Word Length'],
                       y=[stats['avg_word_len']],     marker_color='#f093fb'),
                go.Bar(name='Vocab Diversity ×100',x=['Vocab Diversity'],
                       y=[stats['vocab_diversity']*100], marker_color='#764ba2'),
            ])
            fig_b.update_layout(height=380, title_text="Document Statistics",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.5)',
                font=dict(color='#e0e0e0'))
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("#### Score Breakdown")
        fig_sc = go.Figure(go.Bar(
            x=list(scores.keys()), y=list(scores.values()),
            marker_color=['#667eea','#764ba2','#f093fb','#667eea','#764ba2','#f093fb'],
            text=[f"{v}/100" for v in scores.values()], textposition='outside'))
        fig_sc.update_layout(height=350, yaxis_range=[0,115],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.5)',
            font=dict(color='#e0e0e0'))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Top Keywords bar chart
        if data.get("word_freq"):
            st.markdown("#### 📊 Top Keywords by Frequency")
            wf = data["word_freq"]
            top15 = sorted(wf.items(), key=lambda x: x[1], reverse=True)[:15]
            if top15:
                kw_names = [w for w,_ in top15]
                kw_vals  = [c for _,c in top15]
                fig_kw = go.Figure(go.Bar(
                    x=kw_vals, y=kw_names, orientation='h',
                    marker_color='rgba(118,75,162,0.7)',
                    marker_line_color='#f093fb', marker_line_width=1,
                    text=kw_vals, textposition='outside',
                ))
                fig_kw.update_layout(
                    height=380, yaxis=dict(autorange='reversed'),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.5)',
                    font=dict(color='#e0e0e0'), margin=dict(l=10,r=40,t=10,b=10),
                )
                st.plotly_chart(fig_kw, use_container_width=True)

    # ── Tab 2: Section Summaries (3 sub-tabs) ───────────────────────────
    with tab2:
        st.subheader("Smart Section Summarization")
        FIXED = ["Abstract","Introduction","Methodology","Results","Conclusion"]

        sub1, sub2, sub3 = st.tabs(["📝 Smart Summaries",
                                     "✅ Detected Sections",
                                     "❌ Missing Sections"])
        with sub1:
            st.markdown(f"Summaries generated at **{sum_len}** detail level. "
                        "Sections marked *Inferred* were not explicitly headed.")
            for sec in FIXED:
                content  = merged_secs.get(sec,"").strip()
                detected = sec_det.get(sec, False)
                badge    = "✅ Detected" if detected else "🔍 Inferred"
                with st.expander(f"**{sec}** — {badge}", expanded=False):
                    if content and len(content) > 50:
                        summary = sec_summs.get(sec,{}).get(sum_len,
                                  _heuristic_summary(content, 4))
                        st.markdown(f'<div class="summary-box">{summary}</div>',
                                    unsafe_allow_html=True)
                        if detected:
                            st.success(f"{sec} section explicitly detected.")
                        else:
                            st.info(f"{sec} content inferred from context.")
                    else:
                        st.warning(f"{sec} section content not found.")

        with sub2:
            found = [s for s in FIXED if sec_det.get(s,False)]
            if found:
                for sec in found:
                    with st.expander(f"✅  {sec}", expanded=False):
                        ct = merged_secs.get(sec,"")
                        st.write(ct[:2000] + ("…" if len(ct) > 2000 else ""))
            else:
                st.info("No sections were explicitly detected via headers. "
                        "All content was inferred.")

        with sub3:
            missing = [s for s in FIXED if not sec_det.get(s,False)]
            if missing:
                for sec in missing:
                    st.warning(f"**{sec}** — not found as an explicit heading. "
                               "Content was inferred from context.")
            else:
                st.success("🎉 All five standard academic sections were detected!")

    # ── Tab 3: Issues ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Long Sentences (>30 words)")
        issues = res.get("issues",[])
        if issues:
            for i, s in enumerate(issues[:20]):
                st.warning(f"**{i+1}:** {s}")
        else:
            st.success("No overly long sentences detected!")

    # ── Tab 4: Suggestions ───────────────────────────────────────────────
    with tab4:
        st.subheader("Vocabulary Improvements")
        smap = {
            "very":"extremely", "bad":"adverse", "good":"beneficial",
            "show":"demonstrate", "big":"substantial", "use":"utilize",
            "get":"obtain", "make":"produce", "find":"identify",
            "important":"significant", "this paper":"this study",
            "we do":"we perform", "we see":"we observe",
        }
        tl = full_text.lower()
        found = False
        for sw, cw in smap.items():
            if re.search(r'\b' + re.escape(sw) + r'\b', tl):
                st.info(f"Replace **'{sw}'** → **'{cw}'**")
                found = True
        if not found:
            st.success("Great vocabulary! No common weak words detected.")

    # ── Tab 5: Detailed Metrics ─────────────────────────────────────────
    with tab5:
        st.subheader("Detailed Document Metrics")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**Total Words:** {stats['word_count']}")
            st.markdown(f"**Total Sentences:** {stats['sentence_count']}")
            st.markdown(f"**Avg Sentence Length:** {stats['avg_sentence_len']} words")
        with d2:
            st.markdown(f"**Avg Word Length:** {stats['avg_word_len']} chars")
            st.markdown(f"**Vocab Diversity:** {stats['vocab_diversity']}")
            st.markdown(f"**Complex Word Ratio:** {stats['complex_word_ratio']}")

        st.markdown("#### All Scores")
        mc      = st.columns(6)
        colours = ["#667eea","#764ba2","#f093fb","#667eea","#764ba2","#f093fb"]
        for col,(key,val),colour in zip(mc, scores.items(), colours):
            with col:
                st.markdown(
                    f'<div class="score-card">'
                    f'<div class="score-number" style="color:{colour};">{val}</div>'
                    f'<div class="score-label">{key}</div></div>',
                    unsafe_allow_html=True)

    # ── Tab 6: Sentiment ────────────────────────────────────────────────
    with tab6:
        pol  = res["sentiment"]
        subj = res["subjectivity"]
        m1, m2 = st.columns(2)
        m1.metric("Sentiment Polarity", pol,
                  help="-1 = very negative  |  0 = neutral  |  +1 = very positive")
        m2.metric("Subjectivity Score", subj,
                  help="0 = fully objective  |  1 = fully subjective")
        if pol > 0.05:
            st.success("Positive Tone — constructive, confident writing style.")
        elif pol < -0.05:
            st.warning("Negative/Critical Tone — cautionary or critical language.")
        else:
            st.info("Neutral Tone — objective, formal academic tone.")

        st.markdown("#### Section-wise Sentiment")
        src = {k:v for k,v in merged_secs.items() if v and len(v.strip())>30}
        if src:
            sec_names, sec_pol, sec_subj = [], [], []
            for nm, body in src.items():
                b = TextBlob(body)
                sec_names.append(nm[:30])
                sec_pol.append(round(float(b.sentiment.polarity),4))
                sec_subj.append(round(float(b.sentiment.subjectivity),4))
            bcolours = ['#50c878' if p>0.05 else ('#ff5050' if p<-0.05 else '#aaaaaa')
                        for p in sec_pol]
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Bar(
                name='Polarity', x=sec_names, y=sec_pol,
                marker_color=bcolours,
                text=[str(v) for v in sec_pol], textposition='outside'))
            fig_sec.add_trace(go.Scatter(
                name='Subjectivity', x=sec_names, y=sec_subj,
                mode='lines+markers',
                line=dict(color='#f093fb',width=2),
                marker=dict(size=8,color='#f093fb'), yaxis='y2'))
            fig_sec.update_layout(
                height=400,
                yaxis=dict(title=dict(text='Polarity',font=dict(color='#e0e0e0')),
                           range=[-1,1], tickcolor='#e0e0e0'),
                yaxis2=dict(title=dict(text='Subjectivity',font=dict(color='#f093fb')),
                            range=[0,1], overlaying='y', side='right', tickcolor='#f093fb'),
                legend=dict(font=dict(color='#e0e0e0')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.5)',
                font=dict(color='#e0e0e0'), bargap=0.3,
                shapes=[dict(type='line', x0=-0.5, x1=len(sec_names)-0.5,
                             y0=0, y1=0,
                             line=dict(color='#ffffff',width=1,dash='dot'))])
            st.plotly_chart(fig_sec, use_container_width=True)

            st.markdown("#### Section Sentiment Table")
            for nm, p, s in zip(sec_names, sec_pol, sec_subj):
                tone = "Positive" if p>0.05 else ("Negative" if p<-0.05 else "Neutral")
                obj  = "Objective" if s<0.4 else ("Balanced" if s<0.65 else "Subjective")
                st.markdown(
                    f'<div class="sentiment-row"><b style="color:#f093fb">{nm}</b>'
                    f'&nbsp;|&nbsp; Polarity: <b>{p}</b> &nbsp;{tone}'
                    f'&nbsp;|&nbsp; Subjectivity: <b>{s}</b> ({obj})</div>',
                    unsafe_allow_html=True)
        else:
            st.info("No sections found for per-section sentiment analysis.")

        st.markdown("#### Overall Polarity Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pol,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Sentiment Polarity",'font':{'color':'#f093fb'}},
            number={'font':{'color':'#e0e0e0','size':48}},
            gauge={'axis':{'range':[-1,1],'tickcolor':'#e0e0e0',
                           'tickvals':[-1,-0.5,0,0.5,1]},
                   'bar':{'color':'#764ba2','thickness':0.3},
                   'steps':[{'range':[-1,-0.05],'color':'rgba(255,80,80,0.25)'},
                             {'range':[-0.05,0.05],'color':'rgba(200,200,200,0.15)'},
                             {'range':[0.05,1],'color':'rgba(80,200,120,0.25)'}],
                   'threshold':{'line':{'color':'#f093fb','width':3},
                                'thickness':0.8,'value':pol}}))
        fig_g.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e0e0e0'))
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Tab 7: Cross-Document Q&A ────────────────────────────────────────
    with tab7:
        st.subheader("🔍 Cross-Document Q&A")
        st.markdown("Search across **all** analyzed papers using TF-IDF similarity.")

        question = st.text_input("Ask a question",
                                 placeholder="e.g. What methodology was used?", key="qa_q")
        if st.button("Search Papers", key="qa_search"):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                all_sents, meta = [], []
                for pname, pdata in st.session_state.analyses.items():
                    for sent in TextBlob(pdata["res"].get("full_text","")).sentences:
                        all_sents.append(str(sent))
                        meta.append(pname)
                if not all_sents:
                    st.warning("No text available.")
                else:
                    try:
                        vec  = TfidfVectorizer(stop_words='english')
                        mat  = vec.fit_transform(all_sents)
                        qv   = vec.transform([question])
                        sims = cosine_similarity(qv, mat).flatten()
                        top5 = sims.argsort()[-5:][::-1]
                        st.markdown("**Top relevant excerpts:**")
                        found_any = False
                        for idx in top5:
                            if sims[idx] > 0.05:
                                st.info(f"**{meta[idx]}** (relevance: {sims[idx]:.3f})\n\n"
                                        f"{all_sents[idx]}")
                                found_any = True
                        if not found_any:
                            st.markdown("*No highly relevant sentences found.*")
                    except Exception as e:
                        st.error(f"Search error: {e}")

        st.markdown("---")
        if st.button("💾  Save Current Paper to Library", key="save_paper_btn"):
            abs_sum = sec_summs.get("Abstract",{}).get("Medium","")
            db_save_paper(st.session_state.user_id, fname,
                          abs_sum, scores["Composite"])
            st.success(f"✅ '{fname}' saved to your library!")

    # ── Tab 8: Research Gaps (Feature 9) ────────────────────────────────
    with tab8:
        st.subheader("🧩 Research Gap Analysis")
        st.markdown("Detecting which key research aspects are addressed or missing in this paper.")
        if research_gaps:
            present = [(k, v) for k, v in research_gaps.items() if v]
            missing = [(k, v) for k, v in research_gaps.items() if not v]
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f"**✅ Covered Aspects ({len(present)}/{len(research_gaps)})**")
                for asp, _ in present:
                    st.markdown(f'<span class="gap-present">✔ {asp}</span>', unsafe_allow_html=True)
            with g2:
                st.markdown(f"**⚠️ Potentially Missing ({len(missing)}/{len(research_gaps)})**")
                for asp, _ in missing:
                    st.markdown(f'<span class="gap-missing">✘ {asp}</span>', unsafe_allow_html=True)
            coverage = int(len(present) / len(research_gaps) * 100) if research_gaps else 0
            st.markdown("---")
            st.markdown(f"**Research Coverage Score: {coverage}%**")
            fig_cov = go.Figure(go.Indicator(
                mode="gauge+number", value=coverage,
                domain={'x':[0,1],'y':[0,1]},
                title={'text': "Coverage %", 'font': {'color':'#f093fb'}},
                number={'font':{'color':'#e0e0e0','size':40},'suffix':'%'},
                gauge={'axis':{'range':[0,100],'tickcolor':'#e0e0e0'},
                       'bar':{'color':'#764ba2','thickness':0.3},
                       'steps':[{'range':[0,40],'color':'rgba(255,80,80,.2)'},
                                 {'range':[40,70],'color':'rgba(255,165,0,.2)'},
                                 {'range':[70,100],'color':'rgba(80,200,120,.2)'}]}
            ))
            fig_cov.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#e0e0e0'))
            st.plotly_chart(fig_cov, use_container_width=True)
        else:
            st.info("Research gap data not available for this paper.")

    # ── Tab 9: Highlights (Feature 11) ──────────────────────────────────
    with tab9:
        st.subheader("✨ Key Insight Highlights")
        st.markdown("Automatically extracted sentences containing significant findings and contributions.")
        if highlights:
            weak_words = ['very','really','quite','rather','somewhat','fairly','mostly',
                          'generally','basically','simply','just','only','little','much']
            for sent in highlights:
                # highlight weak words in yellow
                highlighted = sent
                for ww in weak_words:
                    highlighted = re.sub(
                        r'\b' + re.escape(ww) + r'\b',
                        f'<span style="background:rgba(255,200,50,.25);padding:0 2px;border-radius:3px;">{ww}</span>',
                        highlighted, flags=re.IGNORECASE
                    )
                st.markdown(f'<div class="highlight-sentence">{highlighted}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("No significant highlight sentences detected. "
                    "Try a paper with stronger research claims and conclusions.")

# ═══════════════════════════════════════════════════════════════════════════
# SAVED PAPERS VIEW
# ═══════════════════════════════════════════════════════════════════════════
def show_saved_papers():
    st.markdown('<div class="section-header">SAVED PAPERS</div>', unsafe_allow_html=True)
    papers = db_get_saved_papers(st.session_state.user_id)
    if not papers:
        st.info("No saved papers yet. Analyze a paper and save it from the Q&A tab.")
        return
    for p in papers:
        with st.expander(
            f"📄  {p['file_name']}  │  Score: {p['composite_score']}/100  │  {p['saved_time']}",
            expanded=False
        ):
            st.markdown(f"**Abstract Summary:**\n\n{p['abstract_summary']}")
            if st.button(f"🗑️  Delete", key=f"del_{p['id']}"):
                db_delete_saved_paper(p['id'])
                st.success("Deleted.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD HISTORY VIEW
# ═══════════════════════════════════════════════════════════════════════════
def show_upload_history():
    st.markdown('<div class="section-header">UPLOAD HISTORY</div>', unsafe_allow_html=True)
    history = db_get_upload_history(st.session_state.user_id)
    if not history:
        st.info("No uploads recorded yet.")
        return
    df = pd.DataFrame(history)
    rename_map = {"id":"#","file_name":"File","word_count":"Words",
                  "page_count":"Pages","upload_time":"Uploaded At"}
    if "coherence_score" in df.columns:
        rename_map["coherence_score"] = "Coherence"
        df = df.rename(columns=rename_map)
        st.dataframe(df[["#","File","Words","Pages","Coherence","Uploaded At"]],
                     use_container_width=True, hide_index=True)
    else:
        df = df.rename(columns=rename_map)
        st.dataframe(df[["#","File","Words","Pages","Uploaded At"]],
                     use_container_width=True, hide_index=True)
    st.success(f"Total uploads: {len(history)}")

# ═══════════════════════════════════════════════════════════════════════════
# PROFILE VIEW
# ═══════════════════════════════════════════════════════════════════════════
def show_profile():
    """Features 2 & 3: Rich Profile Dashboard + Recent Activity Log."""
    st.markdown('<div class="section-header">USER PROFILE</div>', unsafe_allow_html=True)
    icon = "🎓" if st.session_state.user_role == "teacher" else "📚"
    st.markdown(f"""
    <div class="stats-card" style="max-width:420px;margin:2rem auto;padding:2.5rem;">
        <div style="font-size:4rem;">{icon}</div>
        <div style="color:#f093fb;font-size:1.8rem;font-weight:700;margin-top:1rem;">
            {st.session_state.user_name}</div>
        <div style="color:#e0e0e0;font-size:1rem;margin-top:.5rem;">
            ID: {st.session_state.user_id}</div>
        <div style="color:#888;font-size:.9rem;margin-top:.3rem;">
            Role: {st.session_state.user_role.capitalize()}</div>
    </div>""", unsafe_allow_html=True)

    uploads = db_get_upload_history(st.session_state.user_id)
    saved   = db_get_saved_papers(st.session_state.user_id)
    topics  = db_get_topics(st.session_state.user_id)

    # Compute stats from upload history
    coherence_scores = []
    composite_scores = []
    total_words = 0
    best_score  = 0
    for u in uploads:
        wc = u.get("word_count", 0) or 0
        total_words += wc
        cs = u.get("coherence_score", 0) or 0
        if cs > 0:
            coherence_scores.append(cs)

    # Try to get composite scores from topics/insights if available
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT composite_score FROM paper_insights_store WHERE user_key=?",
                    (st.session_state.user_id,))
        rows = cur.fetchall(); conn.close()
        composite_scores = [r[0] for r in rows if r[0]]
        best_score = max(composite_scores) if composite_scores else 0
    except Exception:
        pass

    avg_composite = round(sum(composite_scores)/len(composite_scores), 1) if composite_scores else 0
    avg_coherence = round(sum(coherence_scores)/len(coherence_scores), 1) if coherence_scores else 0

    # 6 stat cards
    st.markdown("### 📊 Your Stats")
    c1,c2,c3 = st.columns(3)
    c4,c5,c6 = st.columns(3)
    def _stat(col, val, lbl, color):
        col.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{color};">{val}</div>'
                     f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    _stat(c1, len(uploads),    "Papers Analyzed",    "#667eea")
    _stat(c2, len(saved),      "Papers Saved",        "#f093fb")
    _stat(c3, f"{avg_composite}/100", "Avg Composite Score", "#764ba2")
    _stat(c4, f"{avg_coherence}/100", "Avg Coherence Score", "#50c878")
    _stat(c5, f"{total_words:,}", "Total Words Checked", "#ff9f43")
    _stat(c6, f"{best_score}/100", "Best Score",        "#ee5a24")

    st.markdown("<br>", unsafe_allow_html=True)

    # Domain pie chart from topics
    if topics:
        domain_counts = {}
        for t in topics:
            d = t.get("domain","Unknown") or "Unknown"
            domain_counts[d] = domain_counts.get(d,0) + 1
        if domain_counts:
            st.markdown("### 🗺️ Research Domains")
            fig_pie = go.Figure(go.Pie(
                labels=list(domain_counts.keys()),
                values=list(domain_counts.values()),
                hole=0.45,
                marker_colors=['#667eea','#764ba2','#f093fb','#50c878','#ff9f43','#ee5a24'],
            ))
            fig_pie.update_layout(
                height=320, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                legend=dict(font=dict(color='#e0e0e0')),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Feature 3: Recent Activity Log
    st.markdown("### 🕐 Recent Activity")
    if uploads:
        for u in uploads[:10]:
            fname_short = (u.get("file_name","") or "")[:50]
            t = u.get("upload_time","")
            wc = u.get("word_count", 0) or 0
            st.markdown(
                f'<div class="activity-item">📄 Analyzed <b>{fname_short}</b>'
                f' &nbsp;|&nbsp; {wc:,} words'
                f'<span class="activity-time">{t}</span></div>',
                unsafe_allow_html=True)
    else:
        st.info("No activity yet. Start by analyzing a paper!")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP SHELL
# ═══════════════════════════════════════════════════════════════════════════
def show_main_app():
    st.markdown("""
    <div class="header-box">
        <div class="header-title">PAPERIQ</div>
        <div class="header-subtitle">Research Paper Analyzer</div>
    </div>""", unsafe_allow_html=True)

    show_sidebar()

    if st.session_state.user_role == "teacher" and st.session_state.get("show_admin", False):
        show_admin_panel()

    page = st.session_state.sidebar_page
    if   page == "dashboard": show_dashboard()
    elif page == "saved":     show_saved_papers()
    elif page == "history":   show_upload_history()
    elif page == "profile":   show_profile()

# ═══════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════
try:
    if not st.session_state.authenticated:
        pg = st.session_state.get("page")
        if   pg == "signin":          show_role_selector("signin")
        elif pg == "signup":          show_role_selector("signup")
        elif pg == "forgot_password": show_forgot_password()
        else:                         show_landing()
    else:
        show_main_app()
except Exception as e:
    st.error(f"Application error: {e}")
    import traceback
    st.code(traceback.format_exc())