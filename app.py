import streamlit as st

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Legal Document Portal",
                   page_icon="🔐",
                   layout="centered")
# ─────────────────────────────────────────────────────────────

# std. imports
import mysql.connector, smtplib, random, re, warnings, json
from email.mime.text import MIMEText
import fitz
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification
from sklearn.exceptions import ConvergenceWarning

# ───────────── CONFIG (insert your creds) ───────────────────
DB_CONFIG = dict(host="localhost", user="root",
                 password="mysqlvis1510#", database="user_db")

DOCS_DB_CONFIG = dict(host="localhost", user="root",
                      password="mysqlvis1510#", database="docs_db")

SMTP_SENDER = "codetesting1510@gmail.com"
SMTP_PASS   = "azfl jqbf lymo bdpk"
PRIMARY_BLUE = "#4F8BF9"

# ───────────── GLOBAL CSS (incl. nav bar tweaks) ────────────
st.markdown(
    f"""
    <style>
    :root {{
        --primary:{PRIMARY_BLUE}; --text-dark:#353535; --card-radius:20px;
    }}
    .stApp {{
        background:linear-gradient(135deg,var(--primary) 0%,#7fa9ff 100%)!important;
    }}
    footer,#MainMenu{{visibility:hidden;}}

    /* hero */
    .centered-content{{display:flex;flex-direction:column;align-items:center;
        padding-top:8vh;text-align:center;}}
    .centered-content h1{{color:#fff;font-size:3.5rem;font-weight:700;margin-bottom:32px;}}

    /* buttons + inputs */
    .btn-pill button, .stButton>button{{
        all:unset;background:#fff;color:var(--primary);padding:.8em 2.5em;
        border-radius:40px;font-size:1.15rem;font-weight:600;cursor:pointer;
        transition:background .2s;text-align:center;}}
    .stButton>button{{font-size:1.05rem;padding:.8em 2em;}}
    .btn-pill button:hover,.stButton>button:hover{{background:#e9f0ff;}}

    .stTextInput>div>div>input,.stPasswordInput>div>div>input,
    .stSelectbox>div>div{{background:#fff!important;border:2px solid #d4dfff!important;
        border-radius:8px!important;padding:.6em .8em!important;}}

    /* doc card */
    .doc-card{{background:#fff;border-radius:var(--card-radius);padding:1.8rem 2rem;
        margin:1.5rem 0;box-shadow:0 8px 20px rgba(0,0,0,.08);}}
    .doc-card h4,.doc-card h5{{color:var(--text-dark);}}
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────── DB helpers ───────────────────────────────────
@st.cache_resource
def get_user_db():  return mysql.connector.connect(**DB_CONFIG)

@st.cache_resource
def get_docs_db():  return mysql.connector.connect(**DOCS_DB_CONFIG)

def fetch_user(email, pwd=None):
    cur = get_user_db().cursor(dictionary=True)
    if pwd:
        cur.execute("SELECT * FROM sign_up WHERE email=%s AND password=%s",
                    (email, pwd))
    else:
        cur.execute("SELECT * FROM sign_up WHERE email=%s", (email,))
    return cur.fetchone()

def insert_user(n,e,p,r):
    cur=get_user_db().cursor()
    cur.execute("INSERT INTO sign_up (name,email,password,roles) "
                "VALUES(%s,%s,%s,%s)", (n,e,p,r))
    get_user_db().commit()

# ↓  document‑storage helpers
def save_document(email, fname, pdf_bytes, summary, entities, risk):
    """Persist one analysed PDF + results (convert set → list for JSON)."""
    entities_serialisable = {k: sorted(list(v)) for k, v in entities.items()}
    cur = get_docs_db().cursor()
    cur.execute("""
        INSERT INTO documents(user_email,filename,file_blob,summary,entities,risk)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (email, fname, pdf_bytes,
          summary,
          json.dumps(entities_serialisable),
          json.dumps(risk)))
    get_docs_db().commit()

def list_user_docs(email, keyword=""):
    cur=get_docs_db().cursor(dictionary=True)
    sql = """
        SELECT id,filename,summary,entities,risk,uploaded_at
        FROM documents WHERE user_email=%s
    """
    args=[email]
    if keyword:
        sql += " AND filename LIKE %s"
        args.append(f"%{keyword}%")
    sql += " ORDER BY uploaded_at DESC"
    cur.execute(sql, args)
    return cur.fetchall()

def get_blob(doc_id):
    cur=get_docs_db().cursor()
    cur.execute("SELECT file_blob,filename FROM documents WHERE id=%s", (doc_id,))
    return cur.fetchone()

# ───────────── OTP helpers ──────────────────────────────────
def send_otp(email):
    otp=str(random.randint(100000,999999))
    st.session_state.otp[email]=otp
    msg=MIMEText(f"Your OTP is: {otp}")
    msg["Subject"]="OTP Verification"
    msg["From"],msg["To"]=SMTP_SENDER,email
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
        s.login(SMTP_SENDER,SMTP_PASS); s.sendmail(SMTP_SENDER,email,msg.as_string())

def verify_otp(e,code): return st.session_state.otp.get(e)==code

# ───────────── NLP caches ───────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@st.cache_resource(show_spinner=True)
def load_models():
    summ=pipeline("summarization",
            model=AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6"),
            tokenizer=AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6"))
    ner = pipeline("ner",
            model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),
            tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER"),
            aggregation_strategy="simple")
    return summ, ner

# ───────────── Analysis utils ───────────────────────────────
def text_from_pdf_bytes(b):
    return "".join(p.get_text() for p in fitz.open(stream=b, filetype="pdf"))

def split_words(t,n=512,o=50):
    w=t.split(); return [" ".join(w[i:i+n]) for i in range(0,len(w),n-o)]

def summarise(t,s,maxc=300):
    if len(t.split())<=maxc:
        return s(t,min_length=30,do_sample=False)[0]['summary_text']
    return summarise(" ".join(
        s(c,min_length=30,do_sample=False)[0]['summary_text']
        for c in split_words(t,maxc)),s,maxc)

def entities(t,ner):
    d={'PER':set(),'ORG':set(),'LOC':set()}
    for c in split_words(t,256):
        for r in ner(c):
            if r['entity_group'] in d: d[r['entity_group']].add(r['word'])
    return d

def risk_report(txt):
    pat = {
        "Termination":[r"terminate.*without cause",r"termination.*sole discretion",r"may terminate.*any time"],
        "Liability":[r"unlimited liability",r"liable.*any damages",r"not liable.*negligence"],
        "Confidentiality":[r"no confidentiality",r"information.*may be shared",r"not obligated.*confidential"],
        "Jurisdiction":[r"exclusive jurisdiction.*(only|court|state)",r"waive.*jurisdiction",r"court.*only in"],
        "Indemnity":[r"indemnify.*without limit",r"hold harmless.*broadly",r"indemnification.*solely responsible"],
        "Payment":[r"no penalty.*late payment",r"delay.*interest free",r"no consequences.*non-payment"]
    }
    l=txt.lower()
    return {k:("⚠️ Risk Detected" if any(re.search(p,l) for p in v)
               else "✅ No Immediate Risk") for k,v in pat.items()}

# ───────────── router helper ─────────────────────────────────
def goto(page): st.session_state.page=page; st.rerun()

# ───────────── session defaults ─────────────────────────────
for k,d in {"page":"landing","otp":{},"pending":{},"logged":None,"models":None}.items():
    st.session_state.setdefault(k,d)

# ╭── LANDING ────────────────────────────────────────────────╮
if st.session_state.page=="landing":
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown('<h1>🔐 Legal Document Portal</h1>', unsafe_allow_html=True)
    c1,c2 = st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="btn-pill">', unsafe_allow_html=True)
        if st.button("Login"): goto("login")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-pill">', unsafe_allow_html=True)
        if st.button("Sign Up"): goto("signup")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ╭── SIGN‑UP ────────────────────────────────────────────────╮
elif st.session_state.page=="signup":
    st.header("📝 Sign Up")
    n,e,p,r = (st.text_input("Name"),
               st.text_input("Email"),
               st.text_input("Password",type="password"),
               st.selectbox("Role",["lawyer","legal professional","client"]))
    if st.button("Send OTP"):
        if fetch_user(e): st.error("Email already registered.")
        else:
            st.session_state.pending=dict(name=n,email=e,password=p,role=r)
            send_otp(e); st.success("OTP sent."); goto("otp_signup")

# ╭── LOGIN ──────────────────────────────────────────────────╮
elif st.session_state.page=="login":
    st.header("🔓 Login")
    n,e,p = (st.text_input("Name"),
             st.text_input("Email"),
             st.text_input("Password",type="password"))
    if st.button("Send OTP"):
        u=fetch_user(e,p)
        if u and u["name"]==n:
            st.session_state.pending={"email":e}
            send_otp(e); st.success("OTP sent."); goto("otp_login")
        else: st.error("Invalid credentials.")

# ╭── OTP Signup ─────────────────────────────────────────────╮
elif st.session_state.page=="otp_signup":
    st.header("📧 Verify Email – Sign Up")
    code=st.text_input("Enter OTP")
    if st.button("Verify & Register"):
        d=st.session_state.pending
        if verify_otp(d["email"],code):
            insert_user(d["name"],d["email"],d["password"],d["role"])
            st.success("Registered. Please log in."); goto("landing")
        else: st.error("Wrong OTP.")

# ╭── OTP Login ──────────────────────────────────────────────╮
elif st.session_state.page=="otp_login":
    st.header("📧 Verify Email – Login")
    code=st.text_input("Enter OTP")
    if st.button("Verify & Login"):
        e=st.session_state.pending["email"]
        if verify_otp(e,code):
            st.session_state.logged=e; st.success(f"Logged in as {e}"); goto("dashboard")
        else: st.error("Wrong OTP.")

# ╭── DASHBOARD (Upload & History via tabs) ──────────────────╮
elif st.session_state.page=="dashboard":
    st.header("📑 Legal Document Analyzer")
    email = st.session_state.logged
    st.write(f"**Logged in:** {email}")

    # load models lazily
    if st.session_state.models is None:
        with st.spinner("Loading NLP models…"):
            st.session_state.models = load_models()
    summ, ner = st.session_state.models

    upload_tab, hist_tab = st.tabs(["📥 Upload", "📂 History"])

    # ─────────── Upload tab ──────────────────────────────────
    with upload_tab:
        files = st.file_uploader("Upload PDF(s)", type="pdf",
                                 accept_multiple_files=True, key="pdf_uploader")
        if files:
            for f in files:
                pdf_bytes = f.read()
                with st.spinner(f"Analyzing {f.name}…"):
                    txt   = text_from_pdf_bytes(pdf_bytes)
                    sm    = summarise(txt, summ)
                    ent   = entities(txt, ner)
                    risk  = risk_report(txt)

                # save to DB
                save_document(email, f.name, pdf_bytes, sm, ent, risk)

                st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                st.subheader(f"📄 {f.name}")
                st.markdown("##### 📝 Summary"); st.write(sm)
                st.markdown("##### 🔍 Named Entities")
                st.write(f" **PERSON👤 :** {', '.join(ent['PER']) or 'None'}")
                st.write(f" **ORGANISATION🏢 :** {', '.join(ent['ORG']) or 'None'}")
                st.write(f" **LOCATION🌍 :** {', '.join(ent['LOC']) or 'None'}")
                st.markdown("##### 🚨 Risk Report")
                for c,s in risk.items(): st.write(f"- **{c}**: {s}")
                st.markdown("</div>", unsafe_allow_html=True)

    # ─────────── History tab ─────────────────────────────────
    with hist_tab:
        keyword = st.text_input("🔍 Search your documents (by filename)",
                                placeholder="type and press enter",
                                key="history_search")
        docs = list_user_docs(email, keyword)
        if docs:
            for d in docs:
                ent = json.loads(d['entities'])
                risk= json.loads(d['risk'])
                st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                # 👉 HTML‑enabled heading (replaces subheader with unsafe)
                st.markdown(
                    f"<h4 style='margin-bottom:0'>📄 {d['filename']} "
                    f"<sub style='color:gray'>( {d['uploaded_at']:%Y‑%m‑%d %H:%M} )</sub></h4>",
                    unsafe_allow_html=True
                )
                with st.expander("📝 Summary / Entities / Risk"):
                    st.write("**Summary**"); st.write(d['summary'])
                    st.write("**Named Entities**")
                    st.write(f" 👤 {', '.join(ent['PER']) or 'None'}")
                    st.write(f" 🏢 {', '.join(ent['ORG']) or 'None'}")
                    st.write(f" 🌍 {', '.join(ent['LOC']) or 'None'}")
                    st.write("**Risk**")
                    for k,v in risk.items(): st.write(f"- **{k}**: {v}")
                blob, fname = get_blob(d['id'])
                st.download_button("⬇️ Download PDF", blob, file_name=fname,
                                   key=f"dl_{d['id']}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No documents stored yet (or no match for search).")

    if st.button("Logout"):
        st.session_state.logged=None; goto("landing")
