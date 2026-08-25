
import io
import os
import re
import html
import math
import string
import pickle
import base64
import hashlib
import datetime as dt
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


#  CONFIGURATION


APP_TITLE = "Fraudulent Job Posting Detection System"
APP_SHORT = "FJD Analyzer"
MODEL_PATH = "fjd_model.pkl"
DATA_PATH = "fake_job_postings.csv"
DEFAULT_THRESHOLD = 0.35
LINKEDIN_URL = "https://www.linkedin.com/in/waseemvpds"
GITHUB_URL = "https://github.com/waseemvpds"

BRAND_PRIMARY = "#2563eb"
BRAND_DANGER = "#dc2626"
BRAND_SAFE = "#059669"
BRAND_WARN = "#d97706"
BRAND_INK = "#0f172a"

MODEL_METRICS = {
    "Algorithm": "LightGBM (RandomizedSearchCV tuned)",
    "Accuracy": 0.9902,
    "Precision": 0.9792,
    "Recall": 0.8150,
    "F1 Score": 0.8896,
    "ROC AUC": 0.9953,
}

SUSPICIOUS_WORDS = [
    "telegram", "whatsapp", "registration fee", "processing fee",
    "security deposit", "payment required", "wire transfer", "bank account",
    "investment required", "guaranteed income", "apply immediately",
    "limited seats", "immediate joining", "no experience required",
]

EMPLOYMENT_TYPES = ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
EXPERIENCE_LEVELS = [
    "Unknown", "Internship", "Entry level", "Associate", "Mid-Senior level",
    "Director", "Executive", "Not Applicable",
]
EDUCATION_LEVELS = [
    "Unknown", "High School or equivalent", "Certification", "Vocational",
    "Some College Coursework Completed", "Associate Degree", "Bachelor's Degree",
    "Master's Degree", "Doctorate", "Professional", "Some High School Coursework",
    "Vocational - Degree", "Vocational - HS Diploma", "Unspecified",
]
INDUSTRIES_FALLBACK = [
    "Unknown", "Information Technology and Services", "Computer Software",
    "Internet", "Marketing and Advertising", "Education Management",
    "Financial Services", "Hospital & Health Care", "Consumer Services",
    "Telecommunications", "Oil & Energy", "Staffing and Recruiting",
    "Real Estate", "Retail", "Construction", "Accounting", "Insurance",
    "Automotive", "Banking", "Design", "Human Resources", "Logistics and Supply Chain",
    "Management Consulting", "Hospitality", "Pharmaceuticals", "Media Production",
    "Electrical/Electronic Manufacturing", "Leisure, Travel & Tourism",
    "Health, Wellness and Fitness", "Nonprofit Organization Management",
]
FUNCTIONS_FALLBACK = [
    "Unknown", "Information Technology", "Sales", "Engineering", "Customer Service",
    "Marketing", "Administrative", "Health Care Provider", "Design", "Education",
    "Management", "Accounting/Auditing", "Business Development", "Human Resources",
    "Consulting", "Finance", "Production", "Product Management", "Project Management",
    "Quality Assurance", "Research", "Writing/Editing", "Art/Creative", "Legal",
    "Advertising", "Public Relations", "Purchasing", "Distribution", "Other",
]
COUNTRIES_FALLBACK = [
    "Unknown", "US", "GB", "GR", "CA", "DE", "NZ", "IN", "AU", "PH", "NL", "BE",
    "IE", "SG", "ES", "PL", "EG", "FR", "IL", "AE", "IT", "SE", "DK", "PK", "MY",
    "RU", "BR", "ZA", "CN", "JP", "MX", "CH", "AT", "NO", "FI", "PT", "TR", "UA",
]
DEPARTMENTS_FALLBACK = [
    "Unknown", "Sales", "Engineering", "Marketing", "Operations", "IT",
    "Product", "Design", "Finance", "Human Resources", "Customer Service",
    "Development", "Consulting", "Research", "Administration", "Support",
]
TITLES_FALLBACK = [
    "Software Engineer", "Senior Backend Engineer", "Frontend Developer",
    "Full Stack Developer", "Data Analyst", "Data Scientist", "Data Entry Clerk",
    "Machine Learning Engineer", "DevOps Engineer", "QA Engineer",
    "Project Manager", "Product Manager", "Business Analyst",
    "Marketing Executive", "Digital Marketing Specialist", "Sales Executive",
    "Account Manager", "Customer Support Representative", "Graphic Designer",
    "UX Designer", "HR Executive", "Recruiter", "Accountant",
    "Financial Analyst", "Operations Manager", "Administrative Assistant",
    "Content Writer", "Teacher", "Nurse", "Civil Engineer",
    "Mechanical Engineer", "Network Administrator", "Systems Administrator",
    "Web Developer", "Mobile App Developer", "Intern",
]


COMMON_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when
where which while who whom why will with you your yours yourself yourselves
work working works job jobs role roles position positions team teams company companies
client clients customer customers product products service services project projects
experience experienced skill skills skilled require required requirements requires
responsibility responsibilities responsible qualification qualifications qualified
candidate candidates applicant applicants apply application employment employee employer
salary benefit benefits bonus insurance health dental vision paid leave vacation holiday
pension retirement training development career growth opportunity opportunities
manage management manager managing lead leading leader senior junior entry level
degree bachelor master diploma certification certified university college school education
communication communicate written verbal english language fluent
software hardware developer development engineer engineering technical technology
data database analysis analyst analytics report reporting research design designer
marketing sales business finance financial account accounting legal support operations
office remote onsite hybrid location based full part time contract permanent temporary
year years month months week weekly daily hour hours schedule flexible
new must should include including provide provided ensure ensuring maintain maintaining
build building create creating develop developing implement implementing deliver delivery
strong excellent good great ability able knowledge understanding familiar proficiency
plus preferred desirable essential minimum maximum least well also within across
our we you your they their us their company's please contact email phone website
looking seeking hiring join growing global international local national leading
environment culture values mission vision passionate motivated driven dynamic
""".split())

EN_BIGRAMS = set("""
th he in er an re nd at on nt ha es st en ed to it ou ea hi is or ti as te et ng of al de
se le sa si ar ve ra ld ur ce ic ll ri ne ma ns om co ta me no ut el li tr be ch ca cl cr
ro pe di ac ho na ir ge us ol ke fo id ai ee ry ty ul im ap ab ad ag am ei ev ex ec eg ek
ee oo ea ou io ia ie ei ui ua ue ay ey oy aw ew ow bl br dr fl fr gl gr pl pr sc sh sk sl
sm sn sp sq sw tw wh wr sr pt ct ft lt nc nk mp mb nf ns nv rd rg rk rl rm rn rp rs rt rv
ss tt ll mm nn pp rr ff gg dd cc bb ph th ck ng qu ba bi bo bu da do du fa fi ga go gu ja
ki ku la lo lu mi mo mu ni nu pa pi po pu ru sy ty va vi vo wa we wi wo ya yo za zi
ph gh sh ch wh oa ue ui uo eo eu au ao oi
""".split())



#  PAGE SETUP + STYLING

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

#MainMenu, footer { visibility: hidden; }

.fjd-hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    border-radius: 18px;
    padding: 30px 34px;
    color: #fff;
    margin-bottom: 22px;
    box-shadow: 0 18px 40px -22px rgba(30,58,138,.85);
    position: relative;
    overflow: hidden;
}
.fjd-hero:after {
    content:""; position:absolute; right:-60px; top:-60px;
    width:240px; height:240px; border-radius:50%;
    background: rgba(255,255,255,.07);
}
.fjd-hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing:-.5px; }
.fjd-hero p  { font-size: .97rem; opacity: .82; margin: 0; max-width: 780px; line-height:1.55; }
.fjd-footer { display:flex; align-items:center; justify-content:center;
  gap:14px; padding:10px 0 24px; color:#64748b; font-size:.88rem; }
.fjd-footer a { display:inline-flex; opacity:.85; transition:opacity .15s, transform .15s; }
.fjd-footer a:hover { opacity:1; transform:translateY(-1px); }
.fjd-pills { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.fjd-pill {
    background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.22);
    padding: 5px 13px; border-radius: 999px; font-size: .74rem; font-weight: 600;
    letter-spacing:.3px;
}

.fjd-section {
    font-size: .78rem; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: #64748b;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 7px; margin: 26px 0 14px 0;
}

.fjd-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 18px 20px; box-shadow: 0 2px 10px -6px rgba(15,23,42,.25);
}

.fjd-verdict {
    border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 16px;
    box-shadow: 0 20px 40px -26px rgba(0,0,0,.7);
}
.fjd-verdict .lbl { font-size:.74rem; letter-spacing:2px; text-transform:uppercase; opacity:.85; font-weight:700;}
.fjd-verdict .val { font-size:2.4rem; font-weight:800; margin:4px 0 2px 0; letter-spacing:-1px;}
.fjd-verdict .sub { font-size:.92rem; opacity:.9; }
.fjd-fraud { background: linear-gradient(120deg,#7f1d1d 0%,#dc2626 100%); }
.fjd-legit { background: linear-gradient(120deg,#064e3b 0%,#059669 100%); }
.fjd-blocked { background: linear-gradient(120deg,#78350f 0%,#d97706 100%); }

.fjd-metric {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
    padding:14px 16px; text-align:left; height:100%;
}
.fjd-metric .k { font-size:.7rem; text-transform:uppercase; letter-spacing:1px; color:#64748b; font-weight:700;}
.fjd-metric .v { font-size:1.5rem; font-weight:800; color:#0f172a; margin-top:2px; }
.fjd-metric .d { font-size:.75rem; color:#94a3b8; }

.fjd-flag {
    background:#fef2f2; border-left:4px solid #dc2626; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; font-size:.87rem; color:#7f1d1d;
}
.fjd-ok {
    background:#f0fdf4; border-left:4px solid #059669; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; font-size:.87rem; color:#065f46;
}
.fjd-note {
    background:#fffbeb; border-left:4px solid #d97706; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; font-size:.87rem; color:#78350f;
}

.fjd-bar-wrap { background:#e2e8f0; border-radius:999px; height:12px; overflow:hidden; }
.fjd-bar { height:100%; border-radius:999px; }

.fjd-hint { font-size:.75rem; color:#94a3b8; margin-top:-8px; margin-bottom:6px; }
.fjd-hint-bad { font-size:.75rem; color:#dc2626; font-weight:600; margin-top:-8px; margin-bottom:6px;}

.stButton>button {
    border-radius: 10px; font-weight: 650; padding: .55rem 1.1rem; border:1px solid #cbd5e1;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; font-size:.9rem;
}
code, .mono { font-family:'JetBrains Mono', monospace; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


#  MODEL + OPTION LOADING

@st.cache_resource(show_spinner="Loading detection model...")
def load_model(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner=False)
def load_vocabulary(_model) -> set:
    """Real-word vocabulary = common words + the model's own TF-IDF vocabulary."""
    vocab = set(COMMON_WORDS)
    try:
        pre = _model.named_steps["preprocessor"]
        tfidf = pre.named_transformers_["text"]
        for term in tfidf.vocabulary_.keys():
            for tok in str(term).split():
                if len(tok) > 1:
                    vocab.add(tok)
    except Exception:
        pass
    return vocab


@st.cache_data(show_spinner=False)
def load_options(path: str) -> dict:
    """Dropdown options, enriched from the dataset when it is available."""
    opts = {
        "employment_type": list(EMPLOYMENT_TYPES),
        "required_experience": list(EXPERIENCE_LEVELS),
        "required_education": list(EDUCATION_LEVELS),
        "industry": list(INDUSTRIES_FALLBACK),
        "function": list(FUNCTIONS_FALLBACK),
        "country": list(COUNTRIES_FALLBACK),
        "department": list(DEPARTMENTS_FALLBACK),
        "state": ["Unknown"],
        "city": ["Unknown"],
        "titles": list(TITLES_FALLBACK),
        "source": "built-in defaults",
    }
    if not os.path.exists(path):
        return opts
    try:
        df = pd.read_csv(path)
        loc = df["location"].fillna("").str.split(",", n=2, expand=True)
        df["country"] = loc[0].str.strip()
        df["state"] = loc[1].str.strip() if 1 in loc.columns else ""
        df["city"] = loc[2].str.strip() if 2 in loc.columns else ""

        def uniq(col, limit=None):
            if col not in df.columns:
                return []
            vals = (
                df[col].dropna().astype(str).str.strip()
                .replace("", np.nan).dropna().value_counts()
            )
            vals = vals.head(limit) if limit else vals
            return sorted(vals.index.tolist())

        for col in ["employment_type", "required_experience", "required_education",
                    "industry", "function", "country", "department"]:
            found = uniq(col)
            if found:
                opts[col] = ["Unknown"] + [v for v in found if v.lower() != "unknown"]
        for col in ["state", "city"]:
            found = uniq(col, limit=400)
            opts[col] = ["Unknown"] + [v for v in found if v.lower() != "unknown"]
        opts["titles"] = uniq("title", limit=600)
        opts["source"] = f"{os.path.basename(path)} ({len(df):,} postings)"
    except Exception as exc:  # dataset present but unreadable -> keep defaults
        opts["source"] = f"built-in defaults (CSV unreadable: {exc})"
    return opts


#  FEATURE ENGINEERING  (mirrors the notebook exactly)

def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"http\S*#URL_[^\s#]+#", " ", text)
    text = re.sub(r"#URL_[^\s#]+#", " ", text)
    text = re.sub(r"#EMAIL_[^\s#]+#", " ", text)
    text = re.sub(r"#PHONE_[^\s#]+#", " ", text)
    text = text.lower()
    text = re.sub(r"[‘’´`]", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def suspicious_word_hits(text: str) -> list:
    low = str(text).lower()
    hits = []
    for word in SUSPICIOUS_WORDS:
        n = len(re.findall(r"\b" + re.escape(word) + r"\b", low))
        if n:
            hits.append((word, n))
    return hits


def build_feature_row(form: dict) -> pd.DataFrame:
    """Assemble the exact one-row frame the fitted pipeline expects."""
    title = form["title"].strip()
    company_profile = form["company_profile"].strip()
    description = form["description"].strip()
    requirements = form["requirements"].strip()
    benefits = form["benefits"].strip()

    combined = " ".join([title, company_profile, description, requirements, benefits])
    hits = suspicious_word_hits(combined)

    row = {
        "cleaned_text": clean_text(combined),
        "employment_type": form["employment_type"] or "Unknown",
        "required_experience": form["required_experience"] or "Unknown",
        "required_education": form["required_education"] or "Unknown",
        "industry": form["industry"] or "Unknown",
        "function": form["function"] or "Unknown",
        "country": form["country"] or "Unknown",
        "state": form["state"] or "Unknown",
        "city": form["city"] or "Unknown",
        "description_length": len(description),
        "requirements_length": len(requirements),
        "company_profile_length": len(company_profile),
        "benefits_length": len(benefits),
        "suspicious_word_count": sum(n for _, n in hits),
        "telecommuting": int(form["telecommuting"]),
        "has_company_logo": int(form["has_company_logo"]),
        "has_questions": int(form["has_questions"]),
        "has_salary": int(form["has_salary"]),
        "has_company_profile": int(company_profile != ""),
        "has_requirements": int(requirements != ""),
    }
    return pd.DataFrame([row]), hits, combined


#  INPUT QUALITY GATE  (gibberish detection -> warn + block)

def shannon_entropy(text: str) -> float:
    text = re.sub(r"\s+", "", text.lower())
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def bigram_plausibility(text: str) -> float:
    letters = re.sub(r"[^a-z]", "", text.lower())
    if len(letters) < 8:
        return 1.0
    pairs = [letters[i:i + 2] for i in range(len(letters) - 1)]
    good = sum(1 for p in pairs if p in EN_BIGRAMS)
    return good / len(pairs)


def word_consonant_run(word: str) -> int:
    """Longest consonant run *inside a single word* ('y' counts as a vowel)."""
    best = run = 0
    for c in word.lower():
        if c in "aeiouy":
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def token_is_plausible(tok: str, vocab: set) -> bool:
    """A token is acceptable if it is a known word OR it is pronounceable.

    Dictionary membership alone is too strict (proper nouns, jargon, other
    languages), so unknown tokens get a phonotactic check instead. Keyboard
    mash such as 'asdkjahsd' fails both.
    """
    t = tok.lower()
    if t in vocab or t.rstrip("s") in vocab or (t + "s") in vocab:
        return True
    if len(t) <= 2:
        return True
    if len(t) > 22:
        return False
    vr = vowel_ratio(t)
    if not (0.15 <= vr <= 0.72):
        return False
    if max((word_consonant_run(p) for p in t.split("-")), default=0) >= 6:
        return False
    if re.search(r"(.)\1{2,}", t):
        return False
    return bigram_plausibility(t) >= 0.30


def real_word_ratio(text: str, vocab: set):
    """Share of tokens that are real or at least pronounceable words."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return 0.0, 0, []
    known, unknown = 0, []
    for t in tokens:
        if token_is_plausible(t, vocab):
            known += 1
        else:
            unknown.append(t)
    return known / len(tokens), len(tokens), unknown


def vowel_ratio(text: str) -> float:
    letters = re.sub(r"[^a-z]", "", text.lower())
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in "aeiou") / len(letters)


def longest_consonant_run(text: str) -> int:
    """Longest consonant run across the whole field, evaluated per word."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return max((word_consonant_run(w) for w in words), default=0)




def analyse_field(name: str, text: str, vocab: set, min_chars: int,
                  min_words: int, required: bool):
    """Return (list_of_problems, list_of_notes, field_score 0-100)."""
    problems, notes = [], []
    text = (text or "").strip()

    if not text:
        if required:
            problems.append(f"**{name}** is empty — this field is required.")
            return problems, notes, 0.0
        return problems, notes, None  # optional + empty -> not scored

    if len(text) < min_chars:
        problems.append(
            f"**{name}** is too short ({len(text)} characters). "
            f"Please provide at least {min_chars} characters of genuine detail."
        )

    words = re.findall(r"\S+", text)
    if len(words) < min_words:
        problems.append(
            f"**{name}** contains only {len(words)} word(s); at least {min_words} expected."
        )

    alpha_words = [w for w in re.findall(r"[a-zA-Z]+", text)]
    avg_len = np.mean([len(w) for w in alpha_words]) if alpha_words else 0
    if alpha_words and (avg_len < 2.4 or avg_len > 13):
        problems.append(
            f"**{name}** has an implausible average word length ({avg_len:.1f} characters)."
        )

    ratio, n_tokens, unknown = real_word_ratio(text, vocab)
    if n_tokens >= 4 and ratio < 0.60:
        problems.append(
            f"**{name}** appears to be random or non-language text — only "
            f"{ratio * 100:.0f}% of the words are recognisable "
            f"(e.g. {', '.join(unknown[:5])})."
        )
    elif n_tokens >= 4 and ratio < 0.75:
        notes.append(f"**{name}**: {ratio * 100:.0f}% recognisable words — check spelling and jargon.")

    vr = vowel_ratio(text)
    if len(re.sub(r"[^a-z]", "", text.lower())) >= 15 and not (0.20 <= vr <= 0.60):
        problems.append(
            f"**{name}** has an unnatural vowel distribution ({vr * 100:.0f}% vowels), "
            "which is typical of keyboard-mashed text."
        )

    # Consonant runs are only suspicious inside words that are not real words —
    # ordinary English ("strengths", "worldwide") happily contains runs of five.
    bad_runs = [w for w in unknown
                if max((word_consonant_run(p) for p in w.split("-")), default=0) >= 6]
    if bad_runs:
        worst = max(bad_runs, key=lambda w: max(word_consonant_run(p) for p in w.split("-")))
        runlen = max(word_consonant_run(p) for p in worst.split("-"))
        problems.append(
            f"**{name}** contains unpronounceable words such as '{worst}' "
            f"({runlen} consecutive consonants) — likely gibberish."
        )


    if re.search(r"(.)\1{3,}", text):
        problems.append(f"**{name}** contains a character repeated 4+ times in a row.")

    non_alpha = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 20 and non_alpha / len(text) > 0.30:
        problems.append(f"**{name}** is dominated by symbols/punctuation.")

    digits = sum(1 for c in text if c.isdigit())
    if len(text) > 30 and digits / len(text) > 0.40:
        problems.append(f"**{name}** is mostly digits.")

    if len(text) > 60 and len(words) <= 2:
        problems.append(f"**{name}** is one long unbroken token — please write readable sentences.")

    ent = shannon_entropy(text)
    if len(text) >= 40 and (ent < 2.2 or ent > 4.7):
        problems.append(
            f"**{name}** has abnormal character randomness (entropy {ent:.2f}); "
            "natural English text sits between 2.2 and 4.7."
        )

    plaus = bigram_plausibility(text)
    if len(text) >= 40 and plaus < 0.28:
        problems.append(
            f"**{name}** does not follow English letter patterns "
            f"({plaus * 100:.0f}% plausible letter pairs)."
        )

    if len(words) >= 12:
        wc = Counter(w.lower() for w in words)
        top, top_n = wc.most_common(1)[0]
        if top_n / len(words) > 0.35:
            problems.append(f"**{name}** repeats the word '{top}' as filler ({top_n} times).")
        sentences = [s.strip().lower() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 15]
        if len(sentences) >= 3 and len(set(sentences)) <= len(sentences) / 2:
            problems.append(f"**{name}** repeats the same sentence multiple times.")

    # Field score
    score = 100.0
    score -= 26 * len(problems)
    score -= 6 * len(notes)
    if n_tokens >= 4:
        score = min(score, 25 + 75 * min(ratio / 0.85, 1.0))
    return problems, notes, max(0.0, min(100.0, score))


def quality_gate(form: dict, vocab: set):
    """Full pre-model validation. Returns (passed, problems, notes, score, detail)."""
    specs = [
        ("Job title", form["title"], 5, 2, True),
        ("Job description", form["description"], 100, 20, True),
        ("Company profile", form["company_profile"], 20, 5, False),
        ("Requirements", form["requirements"], 20, 4, False),
        ("Benefits", form["benefits"], 20, 3, False),
    ]
    problems, notes, scores, detail = [], [], [], {}
    for name, value, min_c, min_w, required in specs:
        p, n, s = analyse_field(name, value, vocab, min_c, min_w, required)
        problems += p
        notes += n
        detail[name] = {"problems": p, "notes": n, "score": s,
                        "chars": len((value or "").strip())}
        if s is not None:
            scores.append(s)

    combined = " ".join([form["title"], form["company_profile"], form["description"],
                         form["requirements"], form["benefits"]])
    ratio, n_tokens, _ = real_word_ratio(combined, vocab)
    if n_tokens >= 25 and ratio < 0.65:
        problems.append(
            f"**Overall submission** is only {ratio * 100:.0f}% recognisable language. "
            "The posting cannot be assessed."
        )
    if len(re.sub(r"\s+", "", combined)) < 150:
        problems.append(
            "**Overall submission** is too sparse (under 150 characters of content) "
            "for a reliable assessment."
        )

    score = float(np.mean(scores)) if scores else 0.0
    if problems:
        score = min(score, 45.0)
    return (len(problems) == 0), problems, notes, round(score, 1), detail


def field_hint(name: str, text: str, vocab: set, min_chars: int) -> str:
    """Short inline hint rendered under each text area."""
    text = (text or "").strip()
    if not text:
        return ""
    ratio, n_tokens, _ = real_word_ratio(text, vocab)
    if n_tokens >= 4 and ratio < 0.60:
        return (f"<div class='fjd-hint-bad'>⚠ Only {ratio * 100:.0f}% recognisable "
                f"words — this looks like gibberish.</div>")
    return ""


#  RISK EXPLANATION

HARD_SCAM_PATTERNS = [
    (r"registration\s*fee|processing\s*fee|security\s*deposit|refundable\s*(fee|deposit)"
     r"|pay\s*(a\s*)?(small\s*)?(fee|amount|charge)|payment\s*required|advance\s*payment",
     "Asks the applicant to pay money (advance-fee scam pattern)"),
    (r"wire\s*transfer|western\s*union|money\s*gram|moneygram|bitcoin|crypto|usdt|gift\s*card",
     "Requests untraceable payment or transfer method"),
    (r"bank\s*account\s*(details|number|information)|account\s*number|ifsc|routing\s*number"
     r"|upi\s*id|debit\s*card|credit\s*card|cvv|pan\s*card|aadhaar|passport\s*copy"
     r"|social\s*security\s*number|ssn",
     "Requests banking or identity credentials"),
    (r"whatsapp|telegram|signal\s*app|wechat|text\s*me\s*on|dm\s*me",
     "Moves hiring off-platform to a private messaging app"),
]

SOFT_SCAM_PATTERNS = [
    (r"guaranteed\s*(income|money|salary|job|earning)", "Guarantees income or a job"),
    (r"earn\s*(up\s*to\s*)?[$₹€]?\s*\d", "Advertises a specific quick-earning figure"),
    (r"no\s*experience\s*(required|needed)|anyone\s*can\s*(do|apply)|no\s*interview",
     "Claims no experience or no interview is needed"),
    (r"work\s*from\s*home.{0,40}(daily|weekly)\s*payment|daily\s*payout|weekly\s*payout",
     "Promises daily or weekly payouts"),
    (r"urgent(ly)?\s*(hiring|required)|immediate\s*joining|apply\s*immediately"
     r"|limited\s*seats|limited\s*slots|hurry|last\s*date\s*today",
     "Uses artificial urgency or scarcity"),
    (r"only\s*\d{1,2}\s*(hours|hrs)\s*(daily|a\s*day|per\s*day)|part\s*time.{0,20}high\s*(income|salary)",
     "Unrealistic effort-to-pay ratio"),
    (r"registration\s*link|click\s*(the\s*)?link|bit\.ly|tinyurl|forms\.gle",
     "Directs the applicant to an external short link"),
    (r"copy\s*paste\s*(job|work)|ad\s*posting\s*job|typing\s*job|form\s*filling",
     "Classic low-skill scam job format"),
]


def rule_fraud_score(form: dict, hits: list):
    """Auditable scam score in [0, 1] plus reasons and a hard-override flag.

    Returns (score, reasons, forced_fraudulent).
    """
    text = " ".join([form.get("title", ""), form.get("company_profile", ""),
                     form.get("description", ""), form.get("requirements", ""),
                     form.get("benefits", "")]).lower()
    reasons, score, hard = [], 0.0, 0

    for pat, label in HARD_SCAM_PATTERNS:
        if re.search(pat, text):
            hard += 1
            score += 0.42
            reasons.append(("hard", label))
    for pat, label in SOFT_SCAM_PATTERNS:
        if re.search(pat, text):
            score += 0.13
            reasons.append(("soft", label))

    # Structural weaknesses (small individual weight, meaningful in combination)
    struct = 0.0
    if not form.get("company_profile", "").strip():
        struct += 0.09
        reasons.append(("struct", "No company profile provided"))
    if not form.get("has_company_logo"):
        struct += 0.07
        reasons.append(("struct", "No company logo on the posting"))
    if not form.get("requirements", "").strip():
        struct += 0.06
        reasons.append(("struct", "No requirements listed"))
    if len(form.get("description", "").strip()) < 300:
        struct += 0.06
        reasons.append(("struct", "Very short job description"))
    if form.get("country", "Unknown") in ("Unknown", ""):
        struct += 0.04
        reasons.append(("struct", "Location not specified"))
    score += min(struct, 0.22)

    if len(hits) >= 3:
        score += 0.10
        reasons.append(("soft", f"{len(hits)} monitored scam phrases present"))

    score = float(min(score, 0.98))
    n_soft = len([r for r in reasons if r[0] == "soft"])
    forced = hard >= 2 or (hard >= 1 and n_soft >= 2) or n_soft >= 5
    return score, reasons, forced


def risk_factors(form: dict, feats: pd.DataFrame, hits: list) -> list:
    """Human-readable, weighted risk signals (independent of the model)."""
    out = []
    for word, n in hits:
        out.append({"label": f"Suspicious phrase: “{word}”" + (f" ×{n}" if n > 1 else ""),
                    "weight": 18, "kind": "risk"})
    if not form["has_company_logo"]:
        out.append({"label": "No company logo on the posting", "weight": 12, "kind": "risk"})
    if not form["company_profile"].strip():
        out.append({"label": "No company profile provided", "weight": 14, "kind": "risk"})
    if not form["requirements"].strip():
        out.append({"label": "No requirements listed", "weight": 10, "kind": "risk"})
    if not form["has_questions"]:
        out.append({"label": "No screening questions", "weight": 6, "kind": "risk"})
    if not form["has_salary"]:
        out.append({"label": "Salary not disclosed", "weight": 5, "kind": "risk"})
    if len(form["description"].strip()) < 400:
        out.append({"label": "Unusually short job description", "weight": 9, "kind": "risk"})
    if form["required_education"] in ("Unknown", "Unspecified", ""):
        out.append({"label": "Education requirement unspecified", "weight": 4, "kind": "risk"})
    if form["industry"] in ("Unknown", ""):
        out.append({"label": "Industry not specified", "weight": 4, "kind": "risk"})
    if form["country"] in ("Unknown", ""):
        out.append({"label": "Location not specified", "weight": 5, "kind": "risk"})

    if form["has_company_logo"]:
        out.append({"label": "Company logo present", "weight": 8, "kind": "safe"})
    if len(form["company_profile"].strip()) > 200:
        out.append({"label": "Detailed company profile provided", "weight": 10, "kind": "safe"})
    if len(form["description"].strip()) > 800:
        out.append({"label": "Comprehensive job description", "weight": 8, "kind": "safe"})
    if form["has_questions"]:
        out.append({"label": "Screening questions in place", "weight": 6, "kind": "safe"})
    if not hits:
        out.append({"label": "No known scam phrases detected", "weight": 10, "kind": "safe"})
    return out


def top_model_features(model, feats: pd.DataFrame, k: int = 10):
    """Feature importance from LightGBM mapped to readable names, restricted to
    features that are actually active for this submission."""
    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        names = list(pre.get_feature_names_out())
        importances = np.asarray(clf.feature_importances_, dtype=float)
        x = pre.transform(feats)
        x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        contrib = importances * (np.abs(x) > 0)
        order = np.argsort(contrib)[::-1][:k]
        rows = []
        for i in order:
            if contrib[i] <= 0:
                continue
            nm = names[i]
            nm = (nm.replace("text__", "keyword: ")
                    .replace("categorical__", "")
                    .replace("numerical__", "")
                    .replace("binary__", "")
                    .replace("_", " "))
            rows.append((nm, float(contrib[i])))
        total = sum(v for _, v in rows) or 1.0
        return [(n, v / total * 100) for n, v in rows]
    except Exception:
        return []


def risk_band(p: float, threshold: float) -> tuple:
    if p >= max(threshold, 0.75):
        return "CRITICAL", BRAND_DANGER
    if p >= threshold:
        return "HIGH", BRAND_DANGER
    if p >= threshold * 0.6:
        return "ELEVATED", BRAND_WARN
    if p >= threshold * 0.3:
        return "LOW", BRAND_WARN
    return "MINIMAL", BRAND_SAFE



#  CHARTS


def gauge_png(prob: float, threshold: float) -> bytes:
    fig, ax = plt.subplots(figsize=(5.2, 2.9), subplot_kw={"aspect": "equal"})
    theta = np.linspace(np.pi, 0, 200)
    for lo, hi, col in [(0, .33, "#059669"), (.33, .66, "#d97706"), (.66, 1, "#dc2626")]:
        seg = np.linspace(np.pi * (1 - lo), np.pi * (1 - hi), 60)
        ax.plot(np.cos(seg), np.sin(seg), lw=20, color=col, solid_capstyle="butt", alpha=.9)
    ang = np.pi * (1 - prob)
    ax.plot([0, .78 * np.cos(ang)], [0, .78 * np.sin(ang)], lw=3.4, color="#0f172a")
    ax.plot(0, 0, "o", ms=9, color="#0f172a")
    tang = np.pi * (1 - threshold)
    ax.plot([.72 * np.cos(tang), 1.06 * np.cos(tang)],
            [.72 * np.sin(tang), 1.06 * np.sin(tang)], lw=2, ls="--", color="#334155")
    ax.text(0, -.34, f"{prob * 100:.1f}%", ha="center", fontsize=21, fontweight="bold", color="#0f172a")
    ax.text(0, -.56, "fraud probability", ha="center", fontsize=9, color="#64748b")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-.65, 1.25); ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def factors_png(factors: list) -> bytes:
    risks = [f for f in factors if f["kind"] == "risk"][:8]
    safes = [f for f in factors if f["kind"] == "safe"][:6]
    items = risks + safes
    if not items:
        items = [{"label": "No signals", "weight": 1, "kind": "safe"}]
    labels = [f["label"][:44] for f in items][::-1]
    vals = [f["weight"] if f["kind"] == "risk" else -f["weight"] for f in items][::-1]
    cols = ["#dc2626" if v > 0 else "#059669" for v in vals]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, .42 * len(items))))
    ax.barh(labels, vals, color=cols, height=.62)
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("← supports legitimacy    |    supports fraud →", fontsize=8, color="#475569")
    ax.tick_params(labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


#  PDF REPORT

def register_unicode_font() -> tuple:
    """Register DejaVu Sans if available so accented characters render."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
    ]
    bold_candidates = [c.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf") for c in candidates]
    for reg, bold in zip(candidates, bold_candidates):
        if os.path.exists(reg):
            try:
                pdfmetrics.registerFont(TTFont("DejaVuSans", reg))
                if os.path.exists(bold):
                    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))
                    return "DejaVuSans", "DejaVuSans-Bold"
                return "DejaVuSans", "DejaVuSans"
            except Exception:
                break
    return "Helvetica", "Helvetica-Bold"


def esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_pdf(result: dict) -> bytes:
    FONT, FONT_B = register_unicode_font()
    buf = io.BytesIO()

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=FONT, fontSize=9.2,
                          leading=13.4, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1e293b"))
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11,
                           textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("h1", parent=styles["Normal"], fontName=FONT_B, fontSize=19,
                        leading=23, textColor=colors.HexColor("#0f172a"), spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Normal"], fontName=FONT_B, fontSize=11.5,
                        leading=15, textColor=colors.HexColor("#1e3a8a"),
                        spaceBefore=13, spaceAfter=6)
    cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER,
                         textColor=colors.HexColor("#64748b"))

    def page_furniture(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.rect(0, h - 16 * mm, w, 16 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont(FONT_B, 9.5)
        canvas.drawString(18 * mm, h - 10.5 * mm, APP_TITLE.upper())
        canvas.setFont(FONT, 8)
        canvas.drawRightString(w - 18 * mm, h - 10.5 * mm, f"Report {result['report_id']}")
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(.6)
        canvas.line(18 * mm, 15 * mm, w - 18 * mm, 15 * mm)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.setFont(FONT, 7.6)
        canvas.drawString(18 * mm, 11 * mm,
                          "Automated risk assessment · Decision support only · Not a legal determination")
        canvas.drawRightString(w - 18 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=22 * mm, bottomMargin=20 * mm,
                          title=f"Job Posting Risk Report {result['report_id']}",
                          author="Fraudulent Job Posting Detection System",
                          subject="Automated fraud risk assessment of a job advertisement")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_furniture)])

    verdict = result["verdict"]
    v_color = colors.HexColor(BRAND_DANGER if verdict == "FRAUDULENT" else BRAND_SAFE)
    story = []

    # ---- Cover block
    story.append(Paragraph("Job Posting Risk Assessment Report", h1))
    story.append(Paragraph(
        "Machine-learning assessment of the likelihood that the submitted job "
        "advertisement is fraudulent.", small))
    story.append(Spacer(1, 9))

    meta = [
        ["Report identifier", result["report_id"]],
        ["Generated (UTC)", result["ts_utc"]],
        ["Generated (local)", result["ts_local"]],
        ["Job title", result["form"]["title"] or "—"],
        ["Location", result["location"]],
    ]
    t = Table([[Paragraph(f"<b>{esc(k)}</b>", small), Paragraph(esc(v), small)] for k, v in meta],
              colWidths=[45 * mm, doc.width - 45 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ---- Verdict banner
    banner = Table([[
        Paragraph(f"<font color='white' size='8'>ASSESSMENT OUTCOME</font><br/>"
                  f"<font color='white' size='20'><b>{verdict}</b></font><br/>"
                  f"<font color='white' size='9'>Fraud probability "
                  f"{result['prob'] * 100:.2f}% · Risk level {result['band']} · "
                  f"Input quality {result['quality']:.0f}/100</font>", body)
    ]], colWidths=[doc.width])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), v_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(banner)

    # ---- Executive summary
    story.append(Paragraph("1. Executive summary", h2))
    story.append(Paragraph(result["summary"], body))
    story.append(Spacer(1, 8))
    story.append(RLImage(io.BytesIO(result["gauge_png"]), width=88 * mm, height=49 * mm))
    story.append(Paragraph("Figure 1 — Fraud probability against the configured decision threshold.", cap))

    # ---- Risk assessment
    story.append(Paragraph("2. Risk assessment", h2))
    risk_rows = [["#", "Signal", "Direction", "Weight"]]
    for i, f in enumerate(result["factors"], 1):
        risk_rows.append([str(i), f["label"],
                          "Fraud indicator" if f["kind"] == "risk" else "Legitimacy indicator",
                          str(f["weight"])])
    rt = Table([[Paragraph(esc(c), small) for c in r] for r in risk_rows],
               colWidths=[10 * mm, doc.width - 68 * mm, 40 * mm, 18 * mm], repeatRows=1)
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(rt)
    # ---- Submitted details
    story.append(PageBreak())
    story.append(Paragraph("3. Submitted job posting", h2))
    f = result["form"]
    struct = [
        ("Job title", f["title"]), ("Department", f["department"]),
        ("Employment type", f["employment_type"]),
        ("Required experience", f["required_experience"]),
        ("Required education", f["required_education"]),
        ("Industry", f["industry"]), ("Function", f["function"]),
        ("Country", f["country"]), ("State / region", f["state"]), ("City", f["city"]),
        ("Remote / telecommuting", "Yes" if f["telecommuting"] else "No"),
        ("Company logo", "Yes" if f["has_company_logo"] else "No"),
        ("Screening questions", "Yes" if f["has_questions"] else "No"),
        ("Salary disclosed", "Yes" if f["has_salary"] else "No"),
    ]
    stt = Table([[Paragraph(f"<b>{esc(k)}</b>", small), Paragraph(esc(v or '—'), small)]
                 for k, v in struct], colWidths=[52 * mm, doc.width - 52 * mm])
    stt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(stt)

    for label, key in [("Company profile", "company_profile"), ("Job description", "description"),
                       ("Requirements", "requirements"), ("Benefits", "benefits")]:
        story.append(Paragraph(f"3.{[k for k in ['company_profile','description','requirements','benefits']].index(key) + 1} {label}", h2))
        txt = (f[key] or "").strip() or "— not provided —"
        story.append(Paragraph(esc(txt).replace("\n", "<br/>"), body))

    # ---- Author footer
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        f"Prepared by <b>Waseem V P</b> &nbsp;·&nbsp; "
        f"<link href=\"{LINKEDIN_URL}\" color=\"#0A66C2\"><u>LinkedIn</u></link>"
        f" &nbsp;·&nbsp; "
        f"<link href=\"{GITHUB_URL}\" color=\"#181717\"><u>GitHub</u></link>", small))

    doc.build(story)
    return buf.getvalue()


#  PRESETS

PRESETS = {
    "— none —": None,
    "Legitimate example (software engineer)": {
        "title": "Senior Backend Engineer",
        "department": "Engineering",
        "company_profile": (
            "Northbridge Systems is a 240-person B2B software company founded in 2011, "
            "headquartered in Seattle with engineering hubs in Berlin and Bangalore. We build "
            "supply-chain visibility software used by more than 400 manufacturing and logistics "
            "customers worldwide. We are privately held, profitable, and have been recognised "
            "three years running as a great place to work. Our engineering culture emphasises "
            "code review, pair programming, and a sustainable on-call rotation."
        ),
        "description": (
            "We are looking for a Senior Backend Engineer to join our Platform team. You will "
            "design, build and operate the distributed services that ingest and normalise "
            "shipment telemetry from carriers around the world, processing several billion "
            "events per month. Day to day you will write production Python and Go, design "
            "APIs consumed by internal and customer-facing applications, improve the "
            "reliability and observability of existing services, participate in design "
            "reviews, and mentor mid-level engineers on the team. You will work closely with "
            "product management and data science to translate customer needs into resilient "
            "technical solutions, and you will share responsibility for the health of the "
            "systems you build through a weekday-hours on-call rotation."
        ),
        "requirements": (
            "Five or more years of professional backend development experience. Strong "
            "proficiency in Python; working knowledge of Go or a willingness to learn it. "
            "Solid experience with relational databases, message queues such as Kafka, and "
            "containerised deployment on Kubernetes. Demonstrated ability to design and "
            "operate high-throughput distributed systems. Excellent written and verbal "
            "communication skills in English. Bachelor's degree in Computer Science or "
            "equivalent practical experience."
        ),
        "benefits": (
            "Competitive base salary with an annual performance bonus, equity participation, "
            "comprehensive medical, dental and vision insurance for you and your dependants, "
            "a 401(k) plan with company matching, 25 days of paid annual leave plus public "
            "holidays, a yearly professional development budget, and a hybrid working policy "
            "of two days per week in the office."
        ),
        "employment_type": "Full-time", "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Computer Software", "function": "Engineering",
        "country": "US", "state": "WA", "city": "Seattle",
        "telecommuting": False, "has_company_logo": True, "has_questions": True, "has_salary": True,
    },
    "Scam example (advance-fee)": {
        "title": "Data Entry Assistant - Work From Home",
        "department": "",
        "company_profile": "",
        "description": (
            "URGENT HIRING! We need 50 data entry assistants to start work immediately. "
            "No experience required, anyone can do this job. You will earn guaranteed income "
            "of $4500 per month working only 2 hours daily from your home. Limited seats "
            "available so apply immediately. After selection you must pay a small registration "
            "fee for your training kit and ID card. Payment required by wire transfer only. "
            "Contact us on whatsapp or telegram for immediate joining. Send your bank account "
            "details after confirmation so we can process your first weekly salary payment."
        ),
        "requirements": "Basic computer knowledge. Must have laptop and internet connection.",
        "benefits": "Weekly payment, guaranteed income, work from anywhere, immediate joining.",
        "employment_type": "Part-time", "required_experience": "Not Applicable",
        "required_education": "Unspecified", "industry": "Unknown", "function": "Administrative",
        "country": "Unknown", "state": "Unknown", "city": "Unknown",
        "telecommuting": True, "has_company_logo": False, "has_questions": False, "has_salary": False,
    },
    "Gibberish example (should be blocked)": {
        "title": "asdkjh qwkjeh",
        "department": "",
        "company_profile": "zxcvzxcv qwerqwer",
        "description": (
            "asdkjahsd kjahsdkjh askjdhkajshd kqwjehqkwjeh zxmcnbzxmcnb qwoieuqwoieu "
            "askjdhaksjdh mnbvmnbvmnbv poiupoiupoiu lkjhlkjhlkjh zxcvzxcvzxcv "
            "qwertqwertqwert asdfgasdfgasdfg hjklhjklhjkl vbnmvbnmvbnm"
        ),
        "requirements": "kjhkjh lkjlkj mnbmnb",
        "benefits": "qweqwe rtyrty",
        "employment_type": "Full-time", "required_experience": "Entry level",
        "required_education": "Unspecified", "industry": "Unknown", "function": "Unknown",
        "country": "Unknown", "state": "Unknown", "city": "Unknown",
        "telecommuting": False, "has_company_logo": False, "has_questions": False, "has_salary": False,
    },
}


# def split_raw_posting(raw: str) -> dict:
#     """Best-effort split of a pasted advertisement into the form's fields."""
#     out = {"title": "", "company_profile": "", "description": "",
#            "requirements": "", "benefits": ""}
#     if not raw.strip():
#         return out
#     lines = [l.strip() for l in raw.strip().splitlines()]
#     non_empty = [l for l in lines if l]
#     if non_empty:
#         out["title"] = non_empty[0][:120]

#     patterns = {
#         "requirements": r"(requirement|qualification|what you.ll need|skills|who you are|must have)",
#         "benefits": r"(benefit|perks|what we offer|compensation|we offer)",
#         "company_profile": r"(about us|about the company|who we are|company profile|our company)",
#         "description": r"(job description|about the role|responsibilit|what you.ll do|the role)",
#     }
#     current = "description"
#     buckets = {k: [] for k in out}
#     for line in lines[1:]:
#         low = line.lower().strip(" :#-*")
#         matched = None
#         if len(low) <= 60:
#             for key, pat in patterns.items():
#                 if re.search(pat, low):
#                     matched = key
#                     break
#         if matched:
#             current = matched
#             continue
#         buckets[current].append(line)
#     for k in buckets:
#         out[k] = "\n".join(buckets[k]).strip()
#     if not out["description"]:
#         out["description"] = raw.strip()
#     return out

def split_raw_posting(raw: str) -> dict:
    """
    Parse a pasted job posting into the five text fields used by the model.

    Supports formats such as:

    Title : Senior Python Developer
    Company Profile : ...
    Job Description : ...
    Requirements : ...
    Benefits : ...

    It also supports Markdown formatting such as:

    **Title :** Senior Python Developer
    """

    out = {
        "title": "",
        "company_profile": "",
        "description": "",
        "requirements": "",
        "benefits": ""
    }

    if not raw or not raw.strip():
        return out

    lines = [
        line.strip()
        for line in raw.strip().splitlines()
        if line.strip()
    ]

    field_patterns = {
        "title": r"^(?:\*\*)?\s*title\s*:?\s*(?:\*\*)?\s*(?:[:\-→]\s*)?(.*)$",
        "company_profile": r"^(?:\*\*)?\s*company\s*profile\s*:?\s*(?:\*\*)?\s*(?:[:\-→]\s*)?(.*)$",
        "description": r"^(?:\*\*)?\s*(?:job\s*)?description\s*:?\s*(?:\*\*)?\s*(?:[:\-→]\s*)?(.*)$",
        "requirements": r"^(?:\*\*)?\s*requirements?\s*:?\s*(?:\*\*)?\s*(?:[:\-→]\s*)?(.*)$",
        "benefits": r"^(?:\*\*)?\s*benefits?\s*:?\s*(?:\*\*)?\s*(?:[:\-→]\s*)?(.*)$"
    }

    current_field = None

    for line in lines:

        clean_line = line.strip()

        matched_field = None
        matched_value = None

        for field, pattern in field_patterns.items():

            match = re.match(
                pattern,
                clean_line,
                flags=re.IGNORECASE
            )

            if match:
                matched_field = field
                matched_value = match.group(1).strip()

                matched_value = re.sub(
                    r"\*\*",
                    "",
                    matched_value
                ).strip()

                break

        if matched_field:

            current_field = matched_field

            if matched_value:
                out[matched_field] = matched_value

        else:

            if current_field:
                if out[current_field]:
                    out[current_field] += " " + clean_line
                else:
                    out[current_field] = clean_line

    # ---------------------------------------------------------
    # Fallback for completely unlabelled text
    # ---------------------------------------------------------

    if not out["title"] and lines:

        first_line = re.sub(
            r"\*\*",
            "",
            lines[0]
        ).strip()

        out["title"] = first_line[:120]

    if not out["description"]:

        remaining_parts = []

        for field in [
            "company_profile",
            "requirements",
            "benefits"
        ]:

            if out[field]:
                remaining_parts.append(out[field])

        if remaining_parts:
            out["description"] = " ".join(remaining_parts)

        else:
            out["description"] = raw.strip()

    return out


#  SESSION STATE

def init_state():
    defaults = {
        "history": [],
        "result": None,
        "gate": None,
        "prefill": None,
        "threshold": DEFAULT_THRESHOLD,
        # --- game state ---
        "game_score": 0,
        "game_streak": 0,
        "game_best_streak": 0,
        "game_total": 0,
        "game_challenge": None,
        "game_answered": False,
        "game_history": [],
        "game_difficulty": "Medium",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()



#  HEADER


st.markdown(f"""
<div class="fjd-hero">
  <h1>🛡️ {APP_TITLE}</h1>
  <p>Enter the details of a job advertisement and the system will assess the likelihood
     that it is fraudulent, explain the reasoning behind the verdict, and generate a
     formal PDF risk report. Every submission is screened for gibberish and incomplete
     data before it reaches the model.</p>
</div>
""", unsafe_allow_html=True)

# ---- Model load
if not os.path.exists(MODEL_PATH):
    st.error(
        f"### Model file not found\n\n"
        f"`{MODEL_PATH}` is missing from `{os.getcwd()}`.\n\n"
        f"Export it from the notebook with "
        f"`pickle.dump(best_model, open(\"{MODEL_PATH}\", \"wb\"))` and place it "
        f"next to `app.py`, then reload this page."
    )
    st.stop()

try:
    MODEL = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"### Could not load the model\n\n`{exc}`\n\n"
             "This usually means the scikit-learn / LightGBM versions differ from the "
             "environment the model was trained in. Re-pickle the model with the versions "
             "installed here, or install matching versions.")
    st.stop()

VOCAB = load_vocabulary(MODEL)
OPTIONS = load_options(DATA_PATH)



#  SIDEBAR


with st.sidebar:
    st.markdown("### 📊 Model performance")

    for k, v in MODEL_METRICS.items():
        if isinstance(v, float):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:.86rem;"
                f"padding:3px 0;border-bottom:1px dashed #e2e8f0;'>"
                f"<span style='color:#64748b'>{k}</span>"
                f"<b class='mono'>{v:.4f}</b></div>", unsafe_allow_html=True)
    st.caption(f"Algorithm: {MODEL_METRICS['Algorithm']}")

    st.markdown("---")
    st.markdown("### 🗂️ Reference data")
    st.caption(f"Dropdown source: {OPTIONS['source']}")
    st.caption(f"Vocabulary size: {len(VOCAB):,} terms")

    st.markdown("---")
    st.markdown("### 🕘 Session history")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            icon = "🔴" if h["verdict"] == "FRAUDULENT" else "🟢"
            st.markdown(
                f"<div style='font-size:.8rem;padding:4px 0;border-bottom:1px solid #f1f5f9;'>"
                f"{icon} <b>{esc(h['title'][:34])}</b><br/>"
                f"<span style='color:#64748b'>{h['prob'] * 100:.1f}% · {h['time']}</span></div>",
                unsafe_allow_html=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet in this session.")



#  TABS


tab_single, tab_batch, tab_compare, tab_game, tab_stats = st.tabs([
    "🔍  Single analysis", "📦  Batch analysis",
    "⚖️  Compare", "🎮  Real or Fake?", "📊  Dataset stats",
])



#  TAB 1 — SINGLE ANALYSIS

with tab_single:
    left, right = st.columns([1, 1.05], gap="large")

    with left:
        st.markdown("<div class='fjd-section'>Quick start</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            preset_name = st.selectbox("Load an example", list(PRESETS.keys()), key="preset_sel")
        with c2:
            st.write("")
            if st.button("Load example", use_container_width=True):
                if PRESETS[preset_name]:
                    st.session_state.prefill = dict(PRESETS[preset_name])
                    st.session_state.result = None
                    st.session_state.gate = None
                    st.rerun()

        pf = st.session_state.prefill or {}

        def opt_index(key, value, fallback=0):
            lst = OPTIONS[key]
            v = value or "Unknown"
            return lst.index(v) if v in lst else (lst.index("Unknown") if "Unknown" in lst else fallback)

        def searchable_select(label, options, current, placeholder, key):
            """Type-to-search dropdown that also accepts a brand-new value."""
            opts = list(dict.fromkeys([o for o in options if o and str(o) != "nan"]))
            if current and current not in opts:
                opts = [current] + opts
            try:
                return st.selectbox(
                    label, opts, index=(opts.index(current) if current in opts else None),
                    placeholder=placeholder, accept_new_options=True, key=key,
                ) or ""
            except TypeError:
                # Older Streamlit: dropdown of known values + manual entry option
                choices = ["✏️ Type manually…"] + opts
                pick = st.selectbox(label, choices,
                                    index=(choices.index(current) if current in choices else 0),
                                    key=key)
                if pick == "✏️ Type manually…":
                    return st.text_input(f"{label} (manual)", value=current or "",
                                         placeholder=placeholder, key=key + "_manual")
                return pick

        st.markdown("<div class='fjd-section'>Job identity</div>", unsafe_allow_html=True)
        title = searchable_select("Job title *", OPTIONS["titles"], pf.get("title", ""),
                                  "Search or type, e.g. Senior Backend Engineer", "sel_title")

        department = searchable_select("Department", OPTIONS.get("department", []),
                                       pf.get("department", ""),
                                       "Search or type, e.g. Engineering", "sel_department")


        c1, c2 = st.columns(2)
        with c1:
            employment_type = st.selectbox("Employment type", OPTIONS["employment_type"],
                                           index=opt_index("employment_type", pf.get("employment_type")))
            required_education = st.selectbox("Required education", OPTIONS["required_education"],
                                              index=opt_index("required_education", pf.get("required_education")))
            industry = st.selectbox("Industry", OPTIONS["industry"],
                                    index=opt_index("industry", pf.get("industry")))
        with c2:
            required_experience = st.selectbox("Required experience", OPTIONS["required_experience"],
                                               index=opt_index("required_experience", pf.get("required_experience")))
            function = st.selectbox("Function", OPTIONS["function"],
                                    index=opt_index("function", pf.get("function")))
            country = st.selectbox("Country", OPTIONS["country"],
                                   index=opt_index("country", pf.get("country")))

        c1, c2 = st.columns(2)
        with c1:
            state = st.text_input("State / region", value=pf.get("state", "") or "",
                                  placeholder="e.g. CA — searchable suggestions below")
            if state:
                sm = [s for s in OPTIONS["state"] if state.lower() in s.lower()][:6]
                if sm:
                    st.caption("Known values: " + " · ".join(sm))
        with c2:
            city = st.text_input("City", value=pf.get("city", "") or "",
                                 placeholder="e.g. Seattle")
            if city:
                cm = [s for s in OPTIONS["city"] if city.lower() in s.lower()][:6]
                if cm:
                    st.caption("Known values: " + " · ".join(cm))
        state = state.strip() or "Unknown"
        city = city.strip() or "Unknown"

        st.markdown("<div class='fjd-section'>Posting attributes</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            telecommuting = st.toggle("Remote / telecommuting", value=bool(pf.get("telecommuting", False)))
            has_company_logo = st.toggle("Company logo present", value=bool(pf.get("has_company_logo", False)))
        with c2:
            has_questions = st.toggle("Has screening questions", value=bool(pf.get("has_questions", False)))
            has_salary = st.toggle("Salary disclosed", value=bool(pf.get("has_salary", False)))

        st.markdown("<div class='fjd-section'>Posting content</div>", unsafe_allow_html=True)
        company_profile = st.text_area("Company profile", value=pf.get("company_profile", ""),
                                       height=130, placeholder="Who is the employer?")
        st.markdown(field_hint("Company profile", company_profile, VOCAB, 20), unsafe_allow_html=True)

        description = st.text_area("Job description *", value=pf.get("description", ""),
                                   height=210, placeholder="What will the person actually do?")
        st.markdown(field_hint("Job description", description, VOCAB, 100), unsafe_allow_html=True)

        requirements = st.text_area("Requirements", value=pf.get("requirements", ""),
                                    height=140, placeholder="Skills, experience, qualifications")
        st.markdown(field_hint("Requirements", requirements, VOCAB, 20), unsafe_allow_html=True)

        benefits = st.text_area("Benefits", value=pf.get("benefits", ""),
                                height=110, placeholder="Compensation, perks, working conditions")
        st.markdown(field_hint("Benefits", benefits, VOCAB, 20), unsafe_allow_html=True)

        b1, b2 = st.columns([2, 1])
        with b1:
            analyse = st.button("🔍  Analyse this job posting", type="primary",
                                use_container_width=True)
        with b2:
            if st.button("Reset form", use_container_width=True):
                st.session_state.prefill = {}
                st.session_state.result = None
                st.session_state.gate = None
                st.rerun()

    form = {
        "title": title, "department": department, "company_profile": company_profile,
        "description": description, "requirements": requirements, "benefits": benefits,
        "employment_type": employment_type, "required_experience": required_experience,
        "required_education": required_education, "industry": industry, "function": function,
        "country": country, "state": state, "city": city,
        "telecommuting": telecommuting, "has_company_logo": has_company_logo,
        "has_questions": has_questions, "has_salary": has_salary,
    }

    if analyse:
        passed, problems, notes, quality, detail = quality_gate(form, VOCAB)
        st.session_state.gate = {"passed": passed, "problems": problems, "notes": notes,
                                 "quality": quality, "detail": detail}
        if not passed:
            st.session_state.result = None
        else:
            feats, hits, combined = build_feature_row(form)
            model_prob = float(MODEL.predict_proba(feats)[0][1])
            rule_prob, rule_reasons, forced = rule_fraud_score(form, hits)
            prob = 1.0 - (1.0 - model_prob) * (1.0 - rule_prob)
            thr = DEFAULT_THRESHOLD
            verdict = "FRAUDULENT" if (prob >= thr or forced) else "LEGITIMATE"
            if forced:
                prob = max(prob, thr + 0.05)
            band, band_col = risk_band(prob, thr)
            factors = risk_factors(form, feats, hits)
            seen = {f["label"] for f in factors}
            rule_first = []
            for kind, label in rule_reasons:
                if kind in ("hard", "soft") and label not in seen:
                    rule_first.append({"label": label,
                                       "weight": 24 if kind == "hard" else 12,
                                       "kind": "risk"})
                    seen.add(label)
            factors = rule_first + factors
            checklist = [
                ("Company profile", bool(form["company_profile"].strip()),
                 "Genuine employers describe themselves." if not form["company_profile"].strip() else "Provided."),
                ("Company logo", form["has_company_logo"],
                 "Missing logos correlate strongly with fraud." if not form["has_company_logo"] else "Present."),
                ("Requirements", bool(form["requirements"].strip()),
                 "Scams often omit requirements." if not form["requirements"].strip() else "Provided."),
                ("Benefits", bool(form["benefits"].strip()),
                 "Not decisive on its own." if not form["benefits"].strip() else "Provided."),
                ("Screening questions", form["has_questions"],
                 "Real hiring processes usually screen." if not form["has_questions"] else "Present."),
                ("Salary disclosed", form["has_salary"],
                 "Undisclosed pay is common but worth noting." if not form["has_salary"] else "Disclosed."),
                ("Location specified", form["country"] not in ("Unknown", ""),
                 "Vague locations are a warning sign." if form["country"] in ("Unknown", "") else "Specified."),
            ]
            n_items = len([f for f in factors if f["kind"] == "risk"][:8]) + \
                      len([f for f in factors if f["kind"] == "safe"][:6])
            summary = (
                f"The submitted advertisement for the position “{esc(form['title'])}” was assessed "
                f"as <b>{verdict}</b> with a fraud probability of {prob * 100:.2f}%, placing it in "
                f"the <b>{band}</b> risk band. "
                + ("The posting exhibits patterns consistent with fraudulent recruitment activity "
                   "and should not be acted upon without independent verification of the employer."
                   if verdict == "FRAUDULENT" else
                   "No decisive indicators of fraudulent recruitment activity were identified; "
                   "standard due diligence is still recommended before sharing personal data.")
            )
            now = dt.datetime.now()
            rid = "FJD-" + hashlib.sha256(
                (form["title"] + now.isoformat()).encode()).hexdigest()[:8].upper()
            result = {
                "report_id": rid,
                "ts_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "ts_local": now.strftime("%Y-%m-%d %H:%M:%S (local)"),
                "form": form, "prob": prob, "verdict": verdict, "band": band,
                "band_color": band_col, "threshold": thr, "quality": quality,
                "factors": factors, "keyword_hits": hits, "checklist": checklist,
                "model_prob": model_prob, "rule_prob": rule_prob,
                "rule_reasons": rule_reasons,
                "location": ", ".join([v for v in [form["city"], form["state"], form["country"]]
                                       if v and v != "Unknown"]) or "Not specified",
                "summary": summary,
                "top_features": top_model_features(MODEL, feats),
                "gauge_png": gauge_png(prob, thr),
                "factors_png": factors_png(factors),
                "factors_ratio": max(.28, min(.62, .062 * max(n_items, 4))),
            }

            st.session_state.result = result
            st.session_state.history.append({
                "title": form["title"] or "(untitled)", "prob": prob,
                "verdict": verdict, "time": now.strftime("%H:%M:%S"),
            })

    with right:
        st.markdown("<div class='fjd-section'>Assessment</div>", unsafe_allow_html=True)
        gate = st.session_state.gate
        result = st.session_state.result

        if gate and gate["problems"]:
            blocked = not gate["passed"]
            st.markdown(f"""
            <div class="fjd-verdict fjd-blocked">
              <div class="lbl">{'Prediction blocked' if blocked else 'Input warning'}</div>
              <div class="val">{'INVALID INPUT' if blocked else 'LOW QUALITY INPUT'}</div>
              <div class="sub">Input quality score {gate['quality']:.0f}/100 ·
                 {len(gate['problems'])} issue(s) detected</div>
            </div>""", unsafe_allow_html=True)
            if blocked:
                st.markdown(
                    "**The posting was not scored.** Random, placeholder or incomplete text "
                    "cannot be assessed reliably, so no verdict is produced. Fix the issues "
                    "below and analyse again.")
            for p in gate["problems"]:
                st.markdown(f"<div class='fjd-flag'>⛔ {p}</div>", unsafe_allow_html=True)
            for n in gate["notes"]:
                st.markdown(f"<div class='fjd-note'>⚠ {n}</div>", unsafe_allow_html=True)



        elif gate and gate["passed"] and not result:
            st.markdown("<div class='fjd-ok'>✓ Input passed all quality checks.</div>",
                        unsafe_allow_html=True)

        if result:
            cls = "fjd-fraud" if result["verdict"] == "FRAUDULENT" else "fjd-legit"
            icon = "🚨" if result["verdict"] == "FRAUDULENT" else "✅"
            st.markdown(f"""
            <div class="fjd-verdict {cls}">
              <div class="lbl">Assessment outcome</div>
              <div class="val">{icon} {result['verdict']}</div>
              <div class="sub">Fraud probability {result['prob'] * 100:.2f}% ·
                  Risk level {result['band']}</div>
            </div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            for col, k, v, d in [
                (m1, "Fraud probability", f"{result['prob'] * 100:.1f}%", "model + rule signals"),
                (m2, "Risk band", result["band"], "relative to threshold"),
                (m3, "Input quality", f"{result['quality']:.0f}/100", "integrity gate"),
            ]:
                col.markdown(f"<div class='fjd-metric'><div class='k'>{k}</div>"
                             f"<div class='v'>{v}</div><div class='d'>{d}</div></div>",
                             unsafe_allow_html=True)

            st.write("")
            st.image(result["gauge_png"], use_container_width=False, width=430)

            st.markdown("<div class='fjd-section'>Why this verdict</div>", unsafe_allow_html=True)
            risks = [f for f in result["factors"] if f["kind"] == "risk"]
            safes = [f for f in result["factors"] if f["kind"] == "safe"]
            if risks:
                for f in risks:
                    st.markdown(f"<div class='fjd-flag'>▲ {f['label']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='fjd-ok'>No fraud indicators triggered.</div>",
                            unsafe_allow_html=True)
            for f in safes:
                st.markdown(f"<div class='fjd-ok'>▼ {f['label']}</div>", unsafe_allow_html=True)

            st.markdown("<div class='fjd-section'>Export</div>", unsafe_allow_html=True)
            try:
                pdf_bytes = build_pdf(result)
                st.download_button(
                    "📄  Download full PDF risk report",
                    data=pdf_bytes,
                    file_name=f"job_risk_report_{result['report_id']}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
                st.caption(f"Report {result['report_id']} · {len(pdf_bytes) / 1024:.0f} KB")

            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")

        elif not gate:
            st.info("Fill in the form on the left and press **Analyse this job posting**. "
                    "Load one of the examples to see the system in action — including the "
                    "gibberish example, which is deliberately blocked.")



#  TAB 2 — BATCH ANALYSIS

with tab_batch:
    st.markdown("<div class='fjd-section'>Batch scoring</div>", unsafe_allow_html=True)
    st.write("Upload a CSV of postings to score them all at once. Recognised columns: "
             "`title`, `company_profile`, `description`, `requirements`, `benefits`, "
             "`employment_type`, `required_experience`, `required_education`, `industry`, "
             "`function`, `location` (or `country`/`state`/`city`), `telecommuting`, "
             "`has_company_logo`, `has_questions`, `salary_range`. Missing columns are "
             "filled with safe defaults.")

    up = st.file_uploader("CSV file", type=["csv"])
    run_gate = st.checkbox("Flag low-quality rows (they are scored but marked)", value=True)

    if up is not None:
        try:
            bdf = pd.read_csv(up)
            st.success(f"Loaded {len(bdf):,} rows · {len(bdf.columns)} columns")
            st.dataframe(bdf.head(5), use_container_width=True)

            if st.button("▶  Score all rows", type="primary"):
                prog = st.progress(0.0, text="Scoring...")
                rows, quals = [], []
                for i, (_, r) in enumerate(bdf.iterrows()):
                    if "location" in bdf.columns and pd.notna(r.get("location")):
                        parts = str(r["location"]).split(",")
                        ctry = parts[0].strip() if len(parts) > 0 else "Unknown"
                        stt = parts[1].strip() if len(parts) > 1 else "Unknown"
                        cty = parts[2].strip() if len(parts) > 2 else "Unknown"
                    else:
                        ctry = str(r.get("country", "Unknown") or "Unknown")
                        stt = str(r.get("state", "Unknown") or "Unknown")
                        cty = str(r.get("city", "Unknown") or "Unknown")

                    def g(col, default=""):
                        v = r.get(col, default)
                        return default if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)

                    f = {
                        "title": g("title"), "department": g("department"),
                        "company_profile": g("company_profile"), "description": g("description"),
                        "requirements": g("requirements"), "benefits": g("benefits"),
                        "employment_type": g("employment_type", "Unknown") or "Unknown",
                        "required_experience": g("required_experience", "Unknown") or "Unknown",
                        "required_education": g("required_education", "Unknown") or "Unknown",
                        "industry": g("industry", "Unknown") or "Unknown",
                        "function": g("function", "Unknown") or "Unknown",
                        "country": ctry or "Unknown", "state": stt or "Unknown", "city": cty or "Unknown",
                        "telecommuting": bool(int(float(r.get("telecommuting", 0) or 0))),
                        "has_company_logo": bool(int(float(r.get("has_company_logo", 0) or 0))),
                        "has_questions": bool(int(float(r.get("has_questions", 0) or 0))),
                        "has_salary": bool(str(r.get("salary_range", "")).strip()
                                           and str(r.get("salary_range")) != "nan"),
                    }
                    feats, hits, _ = build_feature_row(f)
                    p = float(MODEL.predict_proba(feats)[0][1])
                    q = quality_gate(f, VOCAB)[3] if run_gate else None
                    rows.append(p)
                    quals.append(q)
                    if i % 25 == 0:
                        prog.progress(min(1.0, (i + 1) / len(bdf)), text=f"Scoring {i + 1}/{len(bdf)}")
                prog.progress(1.0, text="Done")

                thr = DEFAULT_THRESHOLD
                out = bdf.copy()
                out["fraud_probability"] = np.round(rows, 4)
                out["prediction"] = np.where(out["fraud_probability"] >= thr, "FRAUDULENT", "LEGITIMATE")
                if run_gate:
                    out["input_quality"] = quals
                    out["quality_flag"] = np.where(
                        pd.Series(quals).fillna(100) < 60, "LOW QUALITY — review manually", "OK")

                n_fraud = int((out["prediction"] == "FRAUDULENT").sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows scored", f"{len(out):,}")
                c2.metric("Flagged fraudulent", f"{n_fraud:,}", f"{n_fraud / max(len(out), 1) * 100:.1f}%")
                c3.metric("Mean fraud probability", f"{np.mean(rows) * 100:.1f}%")

                st.dataframe(out.head(200), use_container_width=True)
                st.download_button("⬇  Download scored CSV",
                                   out.to_csv(index=False).encode("utf-8"),
                                   file_name="scored_job_postings.csv", mime="text/csv",
                                   type="primary")

                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.hist(rows, bins=40, color=BRAND_PRIMARY, alpha=.85)
                ax.axvline(thr, ls="--", color=BRAND_DANGER, label=f"threshold {thr:.2f}")
                ax.set_xlabel("Fraud probability"); ax.set_ylabel("Postings"); ax.legend(fontsize=8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                st.pyplot(fig)
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")


#  HELPER — GAME DATA LOADER


@st.cache_data(show_spinner=False)
def load_game_data(path: str) -> pd.DataFrame:
    """Load the labelled dataset for the Real-or-Fake game."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Keep only rows with a usable description
        df = df[df["description"].fillna("").str.strip().str.len() >= 100].copy()
        df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce").fillna(0).astype(int)
        # Fill text columns
        for col in ["title", "company_profile", "requirements", "benefits", "location"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
            else:
                df[col] = ""
        df = df.reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


GAME_DF = load_game_data(DATA_PATH)


def game_row_to_form(row: pd.Series) -> dict:
    """Convert a dataset row to the same dict shape build_feature_row() expects."""
    loc = str(row.get("location", "")).split(",")
    country = loc[0].strip() if len(loc) > 0 else "Unknown"
    state   = loc[1].strip() if len(loc) > 1 else "Unknown"
    city    = loc[2].strip() if len(loc) > 2 else "Unknown"

    def g(col, default=""):
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)

    return {
        "title":            g("title"),
        "department":       g("department"),
        "company_profile":  g("company_profile"),
        "description":      g("description"),
        "requirements":     g("requirements"),
        "benefits":         g("benefits"),
        "employment_type":  g("employment_type", "Unknown") or "Unknown",
        "required_experience": g("required_experience", "Unknown") or "Unknown",
        "required_education":  g("required_education", "Unknown") or "Unknown",
        "industry":         g("industry", "Unknown") or "Unknown",
        "function":         g("function", "Unknown") or "Unknown",
        "country":          country or "Unknown",
        "state":            state or "Unknown",
        "city":             city or "Unknown",
        "telecommuting":    bool(int(float(row.get("telecommuting", 0) or 0))),
        "has_company_logo": bool(int(float(row.get("has_company_logo", 0) or 0))),
        "has_questions":    bool(int(float(row.get("has_questions", 0) or 0))),
        "has_salary":       bool(str(row.get("salary_range", "")).strip()
                                 and str(row.get("salary_range")) != "nan"),
    }


def pick_challenge(df: pd.DataFrame, difficulty: str, exclude_idx: int = -1) -> pd.Series | None:
    """Pick a random challenge row based on difficulty."""
    if df.empty:
        return None
    pool = df[df.index != exclude_idx].copy()
    if difficulty == "Easy":
        # Easy: flagged by suspicious words OR has very long description (clear legit)
        pool["_sw"] = pool["description"].str.lower().apply(
            lambda t: sum(1 for w in SUSPICIOUS_WORDS if w in t)
        )
        easy = pool[(pool["_sw"] >= 2) | (pool["description"].str.len() >= 1000)]
        pool = easy if len(easy) >= 10 else pool
    elif difficulty == "Hard":
        # Hard: no obvious red flags in title — trickier to spot
        hard_words = ["urgent", "whatsapp", "telegram", "fee", "guaranteed", "immediate"]
        mask = pool["title"].str.lower().apply(
            lambda t: not any(w in t for w in hard_words)
        )
        hard = pool[mask]
        pool = hard if len(hard) >= 10 else pool
    return pool.sample(1).iloc[0]


def streak_multiplier(streak: int) -> float:
    if streak >= 10: return 3.0
    if streak >= 5:  return 2.0
    if streak >= 3:  return 1.5
    return 1.0



#  TAB 3 — COMPARE POSTINGS


with tab_compare:
    st.markdown("<div class='fjd-section'>Side-by-side comparison</div>", unsafe_allow_html=True)
    st.write(
        "Paste two job postings below and the system will analyse them independently, "
        "then compare their fraud risk side by side. You can paste raw text — the system "
        "will auto-split sections."
    )

    cmp_col1, cmp_col2 = st.columns(2, gap="large")
    with cmp_col1:
        st.markdown("**📄 Posting A**")
        raw_a = st.text_area(
            "Paste job posting A", height=280,
            placeholder="Paste full job advertisement text here…",
            key="cmp_raw_a",
        )
    with cmp_col2:
        st.markdown("**📄 Posting B**")
        raw_b = st.text_area(
            "Paste job posting B", height=280,
            placeholder="Paste full job advertisement text here…",
            key="cmp_raw_b",
        )

    cmp_go = st.button("⚖️  Analyse & compare both", type="primary", use_container_width=False)

    if cmp_go:
        if not raw_a.strip() or not raw_b.strip():
            st.warning("Please paste text into both posting fields before comparing.")
        else:
            def _analyse_raw(raw: str):
                """Parse raw text, run quality gate, run model, return result dict."""
                parsed = split_raw_posting(raw)
                form = {
                    **parsed,
                    "department": "",
                    "employment_type": "Unknown",
                    "required_experience": "Unknown",
                    "required_education": "Unknown",
                    "industry": "Unknown",
                    "function": "Unknown",
                    "country": "Unknown",
                    "state": "Unknown",
                    "city": "Unknown",
                    "telecommuting": False,
                    "has_company_logo": False,
                    "has_questions": False,
                    "has_salary": False,
                }
                passed, problems, notes, quality, detail = quality_gate(form, VOCAB)
                if not passed:
                    return {"ok": False, "problems": problems, "form": form}
                feats, hits, combined = build_feature_row(form)
                model_prob = float(MODEL.predict_proba(feats)[0][1])
                rule_prob, rule_reasons, forced = rule_fraud_score(form, hits)
                prob = 1.0 - (1.0 - model_prob) * (1.0 - rule_prob)
                thr = DEFAULT_THRESHOLD
                verdict = "FRAUDULENT" if (prob >= thr or forced) else "LEGITIMATE"
                if forced:
                    prob = max(prob, thr + 0.05)
                band, band_col = risk_band(prob, thr)
                factors = risk_factors(form, feats, hits)
                seen = {f["label"] for f in factors}
                for kind, label in rule_reasons:
                    if kind in ("hard", "soft") and label not in seen:
                        factors = [{"label": label, "weight": 24 if kind == "hard" else 12, "kind": "risk"}] + factors
                        seen.add(label)
                return {
                    "ok": True, "form": form, "prob": prob, "verdict": verdict,
                    "band": band, "band_col": band_col, "factors": factors,
                    "hits": hits, "quality": quality,
                }

            res_a = _analyse_raw(raw_a)
            res_b = _analyse_raw(raw_b)

            # --- Verdict banners ---
            st.markdown("---")
            vc1, vc2 = st.columns(2, gap="large")

            def _verdict_html(res, label):
                if not res["ok"]:
                    return (f"<div class='fjd-verdict fjd-blocked'>"
                            f"<div class='lbl'>{label}</div>"
                            f"<div class='val'>❌ INVALID INPUT</div>"
                            f"<div class='sub'>Fix the issues below and retry</div></div>")
                cls = "fjd-fraud" if res["verdict"] == "FRAUDULENT" else "fjd-legit"
                icon = "🚨" if res["verdict"] == "FRAUDULENT" else "✅"
                return (f"<div class='fjd-verdict {cls}'>"
                        f"<div class='lbl'>{label}</div>"
                        f"<div class='val'>{icon} {res['verdict']}</div>"
                        f"<div class='sub'>Fraud probability {res['prob']*100:.2f}% · "
                        f"Risk level {res['band']} · Quality {res['quality']:.0f}/100</div></div>")

            with vc1:
                st.markdown(_verdict_html(res_a, "Posting A"), unsafe_allow_html=True)
                if not res_a["ok"]:
                    for p in res_a["problems"]:
                        st.markdown(f"<div class='fjd-flag'>⛔ {p}</div>", unsafe_allow_html=True)
            with vc2:
                st.markdown(_verdict_html(res_b, "Posting B"), unsafe_allow_html=True)
                if not res_b["ok"]:
                    for p in res_b["problems"]:
                        st.markdown(f"<div class='fjd-flag'>⛔ {p}</div>", unsafe_allow_html=True)

            if res_a["ok"] and res_b["ok"]:
                # --- Probability comparison bar chart ---
                st.markdown("<div class='fjd-section'>Fraud probability comparison</div>",
                            unsafe_allow_html=True)
                fig_cmp, ax_cmp = plt.subplots(figsize=(7, 2.0))
                labels_cmp = ["Posting A", "Posting B"]
                probs_cmp  = [res_a["prob"] * 100, res_b["prob"] * 100]
                bar_cols   = [BRAND_DANGER if p >= DEFAULT_THRESHOLD * 100 else BRAND_SAFE
                              for p in probs_cmp]
                bars = ax_cmp.barh(labels_cmp, probs_cmp, color=bar_cols, height=0.45)
                ax_cmp.axvline(DEFAULT_THRESHOLD * 100, ls="--", color="#334155", lw=1.2,
                               label=f"Threshold {DEFAULT_THRESHOLD*100:.0f}%")
                for bar, val in zip(bars, probs_cmp):
                    ax_cmp.text(val + 1, bar.get_y() + bar.get_height() / 2,
                                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
                ax_cmp.set_xlim(0, 105)
                ax_cmp.set_xlabel("Fraud probability (%)", fontsize=9)
                ax_cmp.legend(fontsize=8)
                for s in ("top", "right", "left"):
                    ax_cmp.spines[s].set_visible(False)
                st.pyplot(fig_cmp)
                plt.close(fig_cmp)

                # --- Risk factor comparison table ---
                st.markdown("<div class='fjd-section'>Risk factor comparison</div>",
                            unsafe_allow_html=True)

                labels_a = {f["label"] for f in res_a["factors"]}
                labels_b = {f["label"] for f in res_b["factors"]}
                all_labels = sorted(labels_a | labels_b)

                def _kind_emoji(factors_list, lbl):
                    for f in factors_list:
                        if f["label"] == lbl:
                            return ("🔴 Fraud signal" if f["kind"] == "risk"
                                    else "🟢 Legit signal")
                    return "—"

                cmp_rows = [{"Signal": lbl,
                             "Posting A": _kind_emoji(res_a["factors"], lbl),
                             "Posting B": _kind_emoji(res_b["factors"], lbl)}
                            for lbl in all_labels]
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

                # --- Suspicious phrase diff ---
                st.markdown("<div class='fjd-section'>Suspicious phrase diff</div>",
                            unsafe_allow_html=True)
                hits_a_words = {w for w, _ in res_a["hits"]}
                hits_b_words = {w for w, _ in res_b["hits"]}
                only_a = hits_a_words - hits_b_words
                only_b = hits_b_words - hits_a_words
                both   = hits_a_words & hits_b_words

                ph1, ph2, ph3 = st.columns(3)
                with ph1:
                    st.markdown("**Only in A**")
                    if only_a:
                        for w in sorted(only_a):
                            st.markdown(f"<div class='fjd-flag'>⚠ {w}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("None")
                with ph2:
                    st.markdown("**In both**")
                    if both:
                        for w in sorted(both):
                            st.markdown(f"<div class='fjd-note'>⚠ {w}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("None")
                with ph3:
                    st.markdown("**Only in B**")
                    if only_b:
                        for w in sorted(only_b):
                            st.markdown(f"<div class='fjd-flag'>⚠ {w}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("None")



#  TAB 4 — REAL OR FAKE GAME


GAME_CSS = """
<style>
.game-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 26px 28px;
    color: #e2e8f0;
    margin-bottom: 18px;
    box-shadow: 0 12px 40px -16px rgba(0,0,0,0.7);
}
.game-card .gc-title {
    font-size: 1.4rem; font-weight: 800; color: #f1f5f9;
    margin-bottom: 4px;
}
.game-card .gc-meta {
    font-size: .8rem; color: #94a3b8; margin-bottom: 16px;
}
.game-card .gc-section {
    font-size: .7rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #334155; padding-bottom: 5px;
    margin: 14px 0 8px 0;
}
.game-card .gc-body {
    font-size: .9rem; line-height: 1.6; color: #cbd5e1;
    max-height: 220px; overflow-y: auto;
}
.game-score-bar {
    display: flex; gap: 18px; align-items: center;
    background: #0f172a; border: 1px solid #1e3a8a;
    border-radius: 14px; padding: 14px 20px; margin-bottom: 20px;
}
.gs-item { text-align: center; }
.gs-item .gs-val { font-size: 1.6rem; font-weight: 800; color: #38bdf8; }
.gs-item .gs-lbl { font-size: .7rem; color: #64748b; letter-spacing: .8px; text-transform: uppercase; }
.game-correct {
    background: linear-gradient(120deg, #064e3b, #059669);
    border-radius: 14px; padding: 18px 22px; color: #fff;
    font-size: 1.1rem; font-weight: 700; margin-bottom: 14px;
    text-align: center;
}
.game-wrong {
    background: linear-gradient(120deg, #7f1d1d, #dc2626);
    border-radius: 14px; padding: 18px 22px; color: #fff;
    font-size: 1.1rem; font-weight: 700; margin-bottom: 14px;
    text-align: center;
}
.game-streak {
    display: inline-block;
    background: linear-gradient(90deg, #d97706, #f59e0b);
    border-radius: 999px; padding: 4px 14px;
    font-size: .85rem; font-weight: 700; color: #fff; margin-left: 10px;
}
.diff-badge {
    display: inline-block; background: #1e3a8a;
    color: #93c5fd; border-radius: 8px;
    padding: 2px 10px; font-size: .78rem; margin: 2px;
    font-weight: 600;
}
</style>
"""
st.markdown(GAME_CSS, unsafe_allow_html=True)

with tab_game:
    st.markdown("<div class='fjd-section'>🎮 Real or Fake? — Test your fraud-spotting skills</div>",
                unsafe_allow_html=True)
    st.write(
        "The system will pick a random job posting from the real dataset. "
        "Read it carefully and decide — is it **real** or **fake**? "
        "After you guess, the AI will reveal the truth and explain the red flags."
    )

    if GAME_DF.empty:
        st.warning(
            "The game requires `fake_job_postings.csv` to be present in the same folder "
            "as `app.py`. The file was not found or could not be read."
        )
    else:
        # ---- Score bar
        ss = st.session_state
        mult = streak_multiplier(ss.game_streak)
        mult_str = f"×{mult:.0f}" if mult == int(mult) else f"×{mult:.1f}"
        streak_html = (
            f"<span class='game-streak'>🔥 {ss.game_streak} streak {mult_str}</span>"
            if ss.game_streak >= 3 else ""
        )
        st.markdown(f"""
        <div class='game-score-bar'>
            <div class='gs-item'><div class='gs-val'>{ss.game_score}</div><div class='gs-lbl'>Score</div></div>
            <div class='gs-item'><div class='gs-val'>{ss.game_streak}</div><div class='gs-lbl'>Streak</div></div>
            <div class='gs-item'><div class='gs-val'>{ss.game_best_streak}</div><div class='gs-lbl'>Best streak</div></div>
            <div class='gs-item'><div class='gs-val'>{ss.game_total}</div><div class='gs-lbl'>Played</div></div>
            <div class='gs-item'>
                <div class='gs-val'>
                    {f"{int(sum(h['correct'] for h in ss.game_history) / max(len(ss.game_history),1) * 100)}%" if ss.game_history else '—'}
                </div>
                <div class='gs-lbl'>Accuracy</div>
            </div>
            <div style='flex:1'>{streak_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Controls row
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
        with ctrl1:
            diff_pick = st.selectbox(
                "Difficulty",
                ["Easy", "Medium", "Hard"],
                index=["Easy", "Medium", "Hard"].index(ss.game_difficulty),
                key="game_diff_sel",
                help="Easy: obvious red flags | Medium: random | Hard: subtle fakes"
            )
            if diff_pick != ss.game_difficulty:
                ss.game_difficulty = diff_pick
        with ctrl2:
            deal_btn = st.button("🎲  Deal new posting", use_container_width=True,
                                 disabled=bool(ss.game_challenge is not None and not ss.game_answered))
        with ctrl3:
            reset_game = st.button("🔄  Reset score", use_container_width=True)

        if reset_game:
            for k in ["game_score", "game_streak", "game_best_streak",
                      "game_total", "game_challenge", "game_answered", "game_history"]:
                ss[k] = 0 if isinstance(ss[k], int) else (None if k == "game_challenge" else False if k == "game_answered" else [])
            st.rerun()

        if deal_btn:
            cur_idx = ss.game_challenge["_idx"] if ss.game_challenge is not None else -1
            row = pick_challenge(GAME_DF, ss.game_difficulty, exclude_idx=cur_idx)
            if row is not None:
                d = row.to_dict()
                d["_idx"] = int(row.name)
                ss.game_challenge = d
                ss.game_answered = False
                st.rerun()

        # ---- Challenge card
        if ss.game_challenge is not None:
            ch = ss.game_challenge
            is_fraud_truth = int(ch.get("fraudulent", 0)) == 1
            title_disp  = str(ch.get("title", "")).strip() or "(Untitled)"
            loc_disp    = str(ch.get("location", "")).strip() or "Location not specified"
            type_disp   = str(ch.get("employment_type", "")).strip() or ""
            company_disp = str(ch.get("company_profile", "")).strip()
            desc_disp   = str(ch.get("description", "")).strip()
            req_disp    = str(ch.get("requirements", "")).strip()
            ben_disp    = str(ch.get("benefits", "")).strip()
            industry_disp = str(ch.get("industry", "")).strip()

            meta_parts = [p for p in [loc_disp, type_disp, industry_disp] if p and p != "Unknown"]
            meta_str = " · ".join(meta_parts) if meta_parts else "Details not provided"

            # Build card HTML (redact company_profile label to avoid giving away context)
            card_sections = []
            if company_disp:
                card_sections.append(
                    f"<div class='gc-section'>About the company</div>"
                    f"<div class='gc-body'>{html.escape(company_disp[:600])}</div>"
                )
            card_sections.append(
                f"<div class='gc-section'>Job description</div>"
                f"<div class='gc-body'>{html.escape(desc_disp[:800])}</div>"
            )
            if req_disp:
                card_sections.append(
                    f"<div class='gc-section'>Requirements</div>"
                    f"<div class='gc-body'>{html.escape(req_disp[:400])}</div>"
                )
            if ben_disp:
                card_sections.append(
                    f"<div class='gc-section'>Benefits</div>"
                    f"<div class='gc-body'>{html.escape(ben_disp[:300])}</div>"
                )

            st.markdown(f"""
            <div class='game-card'>
                <div class='gc-title'>📋 {html.escape(title_disp)}</div>
                <div class='gc-meta'>{html.escape(meta_str)}</div>
                {''.join(card_sections)}
            </div>
            """, unsafe_allow_html=True)

            # ---- Guess buttons or reveal
            if not ss.game_answered:
                gb1, gb2 = st.columns(2, gap="medium")
                with gb1:
                    guess_real = st.button("✅  This looks REAL", use_container_width=True,
                                           type="secondary", key="guess_real")
                with gb2:
                    guess_fake = st.button("🚨  This looks FAKE", use_container_width=True,
                                           type="primary", key="guess_fake")

                if guess_real or guess_fake:
                    user_said_fraud = guess_fake
                    correct = (user_said_fraud == is_fraud_truth)

                    if correct:
                        mult_now = streak_multiplier(ss.game_streak)
                        points = int(10 * mult_now)
                        ss.game_score += points
                        ss.game_streak += 1
                        ss.game_best_streak = max(ss.game_best_streak, ss.game_streak)
                    else:
                        ss.game_streak = 0
                        points = 0

                    ss.game_total += 1
                    ss.game_answered = True

                    # Run model on this posting for reveal
                    form_ch = game_row_to_form(pd.Series(ch))
                    feats_ch, hits_ch, _ = build_feature_row(form_ch)
                    model_prob_ch = float(MODEL.predict_proba(feats_ch)[0][1])
                    rule_prob_ch, rule_reasons_ch, forced_ch = rule_fraud_score(form_ch, hits_ch)
                    prob_ch = 1.0 - (1.0 - model_prob_ch) * (1.0 - rule_prob_ch)
                    if forced_ch:
                        prob_ch = max(prob_ch, DEFAULT_THRESHOLD + 0.05)

                    ss.game_challenge["_reveal"] = {
                        "correct": correct,
                        "user_said_fraud": user_said_fraud,
                        "is_fraud_truth": is_fraud_truth,
                        "prob": prob_ch,
                        "hits": hits_ch,
                        "rule_reasons": rule_reasons_ch,
                        "points": points,
                    }
                    ss.game_history.append({"correct": correct, "title": title_disp})
                    st.rerun()

            else:
                # ---- Reveal
                rev = ss.game_challenge.get("_reveal", {})
                correct  = rev.get("correct", False)
                is_fraud = rev.get("is_fraud_truth", False)
                prob_rev = rev.get("prob", 0.0)
                hits_rev = rev.get("hits", [])
                rule_rev = rev.get("rule_reasons", [])
                pts      = rev.get("points", 0)

                if correct:
                    bonus_txt = f" +{pts} pts" + (" 🔥" if ss.game_streak >= 3 else "")
                    st.markdown(
                        f"<div class='game-correct'>🎉 Correct! That posting was "
                        f"{'FRAUDULENT 🚨' if is_fraud else 'REAL ✅'}{bonus_txt}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='game-wrong'>❌ Wrong! That posting was actually "
                        f"{'FRAUDULENT 🚨' if is_fraud else 'REAL ✅'} — better luck next time!</div>",
                        unsafe_allow_html=True
                    )

                # AI analysis reveal
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**🤖 AI Fraud Probability**")
                    truth_label = "FRAUDULENT" if is_fraud else "LEGITIMATE"
                    truth_col   = BRAND_DANGER if is_fraud else BRAND_SAFE
                    st.markdown(
                        f"<div class='fjd-metric'>"
                        f"<div class='k'>Model verdict</div>"
                        f"<div class='v' style='color:{truth_col}'>{truth_label}</div>"
                        f"<div class='d'>AI probability: {prob_rev*100:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with r2:
                    st.markdown("**📌 Red flags detected**")
                    if hits_rev:
                        for word, count in hits_rev[:5]:
                            st.markdown(
                                f"<span class='diff-badge'>⚠ {word}" +
                                (f" ×{count}" if count > 1 else "") + "</span>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("No obvious red flags in scam keyword list.")

                if rule_rev:
                    st.markdown("**🔎 Why the AI flagged it:**")
                    hard_reasons = [(k, l) for k, l in rule_rev if k == "hard"]
                    soft_reasons = [(k, l) for k, l in rule_rev if k == "soft"]
                    shown = 0
                    for _, label in (hard_reasons + soft_reasons)[:4]:
                        st.markdown(f"<div class='fjd-flag'>▲ {label}</div>", unsafe_allow_html=True)
                        shown += 1
                    struct_reasons = [(k, l) for k, l in rule_rev if k == "struct"]
                    for _, label in struct_reasons[:2]:
                        st.markdown(f"<div class='fjd-note'>● {label}</div>", unsafe_allow_html=True)

                st.write("")
                next_btn = st.button("Next challenge →", type="primary", key="next_challenge")
                if next_btn:
                    row = pick_challenge(GAME_DF, ss.game_difficulty,
                                        exclude_idx=ss.game_challenge.get("_idx", -1))
                    if row is not None:
                        d = row.to_dict()
                        d["_idx"] = int(row.name)
                        ss.game_challenge = d
                        ss.game_answered = False
                        st.rerun()

        elif ss.game_challenge is None:
            st.info(
                "Press **🎲 Deal new posting** to get your first challenge! "
                "Try to spot the fraud patterns before the AI reveals the answer."
            )

        # ---- Recent history panel
        if ss.game_history:
            with st.expander(f"📜 Round history ({len(ss.game_history)} played)", expanded=False):
                for h in reversed(ss.game_history[-15:]):
                    icon = "✅" if h["correct"] else "❌"
                    st.caption(f"{icon} {h['title'][:60]}")



#  TAB 5 — DATASET STATS


with tab_stats:
    st.markdown("<div class='fjd-section'>Dataset analytics</div>", unsafe_allow_html=True)

    if GAME_DF.empty:
        st.warning(
            "`fake_job_postings.csv` is required for dataset statistics. "
            "Place it in the same folder as `app.py` and reload."
        )
    else:
        sdf = GAME_DF.copy()

        # Parse location
        if "location" in sdf.columns:
            loc_split = sdf["location"].str.split(",", n=2, expand=True)
            sdf["_country"] = loc_split[0].str.strip().replace("", "Unknown").fillna("Unknown")
        elif "country" in sdf.columns:
            sdf["_country"] = sdf["country"].fillna("Unknown")
        else:
            sdf["_country"] = "Unknown"

        total    = len(sdf)
        n_fraud  = int(sdf["fraudulent"].sum())
        pct_fr   = n_fraud / max(total, 1) * 100

        # --- Overview metrics
        st.markdown("<div class='fjd-section'>Overview</div>", unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        o1.markdown(f"<div class='fjd-metric'><div class='k'>Total postings</div>"
                    f"<div class='v'>{total:,}</div><div class='d'>in dataset</div></div>",
                    unsafe_allow_html=True)
        o2.markdown(f"<div class='fjd-metric'><div class='k'>Fraudulent</div>"
                    f"<div class='v' style='color:{BRAND_DANGER}'>{n_fraud:,}</div>"
                    f"<div class='d'>{pct_fr:.1f}% of total</div></div>",
                    unsafe_allow_html=True)
        o3.markdown(f"<div class='fjd-metric'><div class='k'>Legitimate</div>"
                    f"<div class='v' style='color:{BRAND_SAFE}'>{total - n_fraud:,}</div>"
                    f"<div class='d'>{100-pct_fr:.1f}% of total</div></div>",
                    unsafe_allow_html=True)
        top_country = sdf["_country"].value_counts().index[0] if len(sdf) > 0 else "N/A"
        o4.markdown(f"<div class='fjd-metric'><div class='k'>Top country</div>"
                    f"<div class='v'>{top_country}</div><div class='d'>by posting count</div></div>",
                    unsafe_allow_html=True)

        st.write("")

        # --- Fraud by industry (horizontal bar)
        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown("**📊 Fraud rate by industry (top 12)**")
            if "industry" in sdf.columns:
                ind_g = (
                    sdf.groupby("industry")["fraudulent"]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "fraud", "count": "total"})
                )
                ind_g = ind_g[ind_g["total"] >= 10].copy()
                ind_g["rate"] = ind_g["fraud"] / ind_g["total"]
                ind_g = ind_g.nlargest(12, "total").sort_values("rate")
                fig_ind, ax_ind = plt.subplots(figsize=(5.5, 3.8))
                bar_colors_ind = [BRAND_DANGER if r > 0.15 else BRAND_SAFE
                                  for r in ind_g["rate"]]
                ax_ind.barh(ind_g.index, ind_g["rate"] * 100, color=bar_colors_ind, height=0.55)
                ax_ind.set_xlabel("Fraud rate (%)", fontsize=8)
                ax_ind.tick_params(labelsize=7)
                for s in ("top", "right", "left"):
                    ax_ind.spines[s].set_visible(False)
                ax_ind.grid(axis="x", alpha=0.2)
                st.pyplot(fig_ind)
                plt.close(fig_ind)
            else:
                st.caption("Industry column not found in dataset.")

        with chart2:
            st.markdown("**🌍 Fraud rate by country (top 10)**")
            ctry_g = (
                sdf.groupby("_country")["fraudulent"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "fraud", "count": "total"})
            )
            ctry_g = ctry_g[ctry_g["total"] >= 20].copy()
            ctry_g["rate"] = ctry_g["fraud"] / ctry_g["total"]
            ctry_g = ctry_g.nlargest(10, "total").sort_values("rate")
            fig_ctry, ax_ctry = plt.subplots(figsize=(5.5, 3.8))
            bar_colors_ctry = [BRAND_DANGER if r > 0.08 else BRAND_SAFE
                               for r in ctry_g["rate"]]
            ax_ctry.barh(ctry_g.index, ctry_g["rate"] * 100, color=bar_colors_ctry, height=0.55)
            ax_ctry.set_xlabel("Fraud rate (%)", fontsize=8)
            ax_ctry.tick_params(labelsize=8)
            for s in ("top", "right", "left"):
                ax_ctry.spines[s].set_visible(False)
            ax_ctry.grid(axis="x", alpha=0.2)
            st.pyplot(fig_ctry)
            plt.close(fig_ctry)

        # --- Description length distribution
        st.markdown("**📏 Description length: real vs fake**")
        sdf["_desc_len"] = sdf["description"].str.len()
        fig_len, ax_len = plt.subplots(figsize=(8, 2.6))
        legit_lens = sdf[sdf["fraudulent"] == 0]["_desc_len"]
        fraud_lens = sdf[sdf["fraudulent"] == 1]["_desc_len"]
        bins_len = np.linspace(0, min(sdf["_desc_len"].quantile(0.98), 8000), 50)
        ax_len.hist(legit_lens.clip(upper=8000), bins=bins_len, alpha=0.65,
                    color=BRAND_SAFE, label="Legitimate", density=True)
        ax_len.hist(fraud_lens.clip(upper=8000), bins=bins_len, alpha=0.65,
                    color=BRAND_DANGER, label="Fraudulent", density=True)
        ax_len.set_xlabel("Description character length", fontsize=9)
        ax_len.set_ylabel("Density", fontsize=9)
        ax_len.legend(fontsize=8)
        for s in ("top", "right"):
            ax_len.spines[s].set_visible(False)
        ax_len.grid(axis="y", alpha=0.2)
        st.pyplot(fig_len)
        plt.close(fig_len)

        # --- Top suspicious phrases in fraudulent postings
        st.markdown("**🚨 Suspicious phrase frequency in fraudulent postings**")
        if n_fraud > 0:
            fraud_text = " ".join(
                sdf[sdf["fraudulent"] == 1]["description"].fillna("").str.lower().tolist()
            )
            phrase_counts = []
            for phrase in SUSPICIOUS_WORDS:
                cnt = len(re.findall(r"\b" + re.escape(phrase) + r"\b", fraud_text))
                phrase_counts.append((phrase, cnt))
            phrase_df = pd.DataFrame(phrase_counts, columns=["Phrase", "Occurrences"])\
                          .sort_values("Occurrences", ascending=True)
            fig_ph, ax_ph = plt.subplots(figsize=(8, max(2.4, 0.35 * len(phrase_df))))
            ax_ph.barh(phrase_df["Phrase"], phrase_df["Occurrences"],
                       color=BRAND_DANGER, height=0.55, alpha=0.85)
            ax_ph.set_xlabel("Occurrences in fraudulent postings", fontsize=9)
            ax_ph.tick_params(labelsize=8)
            for s in ("top", "right", "left"):
                ax_ph.spines[s].set_visible(False)
            ax_ph.grid(axis="x", alpha=0.2)
            st.pyplot(fig_ph)
            plt.close(fig_ph)
        else:
            st.caption("No fraudulent postings found in the loaded dataset.")

        # --- Employment type breakdown
        st.markdown("**🕐 Fraud by employment type**")
        if "employment_type" in sdf.columns:
            et_g = (
                sdf.groupby("employment_type")["fraudulent"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "fraud", "count": "total"})
            )
            et_g = et_g[et_g["total"] >= 5].copy()
            et_g["legit"] = et_g["total"] - et_g["fraud"]
            et_g = et_g.sort_values("total", ascending=False).head(8)
            fig_et, ax_et = plt.subplots(figsize=(8, 2.8))
            x_et = np.arange(len(et_g))
            w_et = 0.38
            ax_et.bar(x_et - w_et/2, et_g["legit"], width=w_et, color=BRAND_SAFE,
                      label="Legitimate", alpha=0.85)
            ax_et.bar(x_et + w_et/2, et_g["fraud"], width=w_et, color=BRAND_DANGER,
                      label="Fraudulent", alpha=0.85)
            ax_et.set_xticks(x_et)
            ax_et.set_xticklabels(et_g.index, rotation=18, ha="right", fontsize=8)
            ax_et.set_ylabel("Postings", fontsize=9)
            ax_et.legend(fontsize=8)
            for s in ("top", "right"):
                ax_et.spines[s].set_visible(False)
            ax_et.grid(axis="y", alpha=0.2)
            st.pyplot(fig_et)
            plt.close(fig_et)
        else:
            st.caption("employment_type column not found.")



#  FOOTER


st.markdown("---")
st.markdown(f"""
<div class="fjd-footer">
  <span>Built by <b>Waseem V P</b></span>
  <a href="{LINKEDIN_URL}" target="_blank" rel="noopener" title="LinkedIn">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="#0A66C2" xmlns="http://www.w3.org/2000/svg">
      <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.13 1.45-2.13 2.94v5.67H9.36V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.55V9h3.57v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/>
    </svg>
  </a>
  <a href="{GITHUB_URL}" target="_blank" rel="noopener" title="GitHub">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="#181717" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 .3a12 12 0 0 0-3.79 23.4c.6.1.82-.26.82-.57v-2c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.2.08 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.34-5.47-5.95 0-1.32.47-2.4 1.24-3.24-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.65.25 2.87.12 3.17.77.84 1.24 1.92 1.24 3.24 0 4.62-2.81 5.64-5.49 5.94.43.37.81 1.1.81 2.22v3.29c0 .32.21.69.83.57A12 12 0 0 0 12 .3z"/>
    </svg>
  </a>
</div>
""", unsafe_allow_html=True)

