#!/usr/bin/env python3
"""
omnixcore.py — OmniX AI v9.1 Core Engine (Offline, Streamlit-free)

Fully local. No Streamlit dependency anywhere. Designed to be driven by a
plain Python state object (see AppState in omnix_app.py) rather than
st.session_state. Every function that used to read/write st.session_state
now takes an explicit `state` argument (duck-typed — needs at least the
attributes used below: kb, active_project, use_sem_mem, dreams_enabled,
auto_improve_enabled, deep_research_enabled, thinking_mode).

Everything talks to Ollama on localhost, so as long as `ollama serve` is
running locally, none of this touches the network.
"""

import sys, io, json, os, time, threading, queue, sqlite3, shutil, socket, struct
import webbrowser, base64, asyncio, tempfile, platform, re, textwrap, hashlib, logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from math import log, sqrt
from typing import Optional, Callable, Any, Union
import ast, operator as op, math, subprocess, traceback, requests, shlex

# ── LOGGING ────────────────────────────────────────────────────
logging.basicConfig(
    filename="omnix.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("OmniX")

# ── Config ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "ollama_base": "http://localhost:11434",
    "db_file": "omnix.db",
    "chromadb_path": "omnix_vector_db",
    "embedding_model": "nomic-embed-text",
    "watchdog_interval": 30,
    "sandbox_timeout": 10,
    "auto_compress_threshold": 12,
    "telemetry_file": "omnix_telemetry.json",
    "task_file": "omnix_tasks.json",
    "graph_file": "omnix_graph.json",
    "world_file": "omnix_world.json",
    "cache_max": 300, "cache_ttl": 7200, "short_ctx": 10,
    "max_memory": 300, "sem_top_k": 6,
    "ollama_retries": 3, "ollama_backoff": 2,
}
def load_config(path="config.yaml"):
    cfg = DEFAULT_CONFIG.copy()
    if Path(path).exists():
        try:
            import yaml
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    cfg.update(loaded)
        except Exception:
            pass
    return cfg
CFG = load_config()
OLLAMA_BASE = CFG["ollama_base"]; DB_FILE = Path(CFG["db_file"])
VECTOR_DB_PATH = Path(CFG["chromadb_path"]); EMBEDDING_MODEL = CFG["embedding_model"]
WATCHDOG_INTERVAL = CFG["watchdog_interval"]; SANDBOX_SEC = CFG["sandbox_timeout"]
TELEMETRY_FILE = Path(CFG["telemetry_file"]); TASK_FILE = Path(CFG["task_file"])
GRAPH_FILE = Path(CFG["graph_file"]); WORLD_FILE = Path(CFG["world_file"])
SEM_TOP_K = CFG["sem_top_k"]; MAX_MEM = CFG["max_memory"]; SHORT_CTX = CFG["short_ctx"]
_CACHE_MAX = CFG["cache_max"]; _CACHE_TTL = CFG["cache_ttl"]
OLLAMA_RETRIES = CFG["ollama_retries"]; OLLAMA_BACKOFF = CFG["ollama_backoff"]
PROJ_FILE = Path("omnix_projects.json"); AGENTS_FILE = Path("omnix_agents.json")
NOTES_FILE = Path("omnix_notes.json"); KB_FILE = Path("omnix_kb.json")
WATCHDOG_LOG = Path("omnix_watchdog.log")

# ── Lazy imports (all optional, all local) ───────────────────
_IMPORT_CACHE = {}
def lazy_import(module_name, attribute=None):
    key = (module_name, attribute)
    if key not in _IMPORT_CACHE:
        try:
            mod = __import__(module_name, fromlist=[attribute]) if attribute else __import__(module_name)
            _IMPORT_CACHE[key] = getattr(mod, attribute) if attribute else mod
        except ImportError:
            _IMPORT_CACHE[key] = None
    return _IMPORT_CACHE[key]
def _psutil(): return lazy_import("psutil")
def _pypdf(): return lazy_import("pypdf")
def _docx(): return lazy_import("docx")
def _fpdf(): return lazy_import("fpdf", "FPDF")
def _pyttsx3(): return lazy_import("pyttsx3")
def _edge_tts(): return lazy_import("edge_tts")
def _pygame(): return lazy_import("pygame")
def _PIL_Image(): return lazy_import("PIL", "Image")
def _PIL_ImageGrab(): return lazy_import("PIL", "ImageGrab")
def _speech_recognition(): return lazy_import("speech_recognition")
def _pyperclip(): return lazy_import("pyperclip")
def _matplotlib(): return lazy_import("matplotlib")
def _plt(): return lazy_import("matplotlib.pyplot")
def _chromadb(): return lazy_import("chromadb")
def _BeautifulSoup(): return lazy_import("bs4", "BeautifulSoup")
def _pvporcupine(): return lazy_import("pvporcupine")
def _pyaudio(): return lazy_import("pyaudio")
def _cv2(): return lazy_import("cv2")
def _sympy(): return lazy_import("sympy")
def _numpy(): return lazy_import("numpy")

HAS_PSUTIL = _psutil() is not None; HAS_PYPDF = _pypdf() is not None; HAS_DOCX = _docx() is not None
HAS_FPDF = _fpdf() is not None; HAS_PYTTSX3 = _pyttsx3() is not None; HAS_EDGE = _edge_tts() is not None
HAS_PYGAME = _pygame() is not None; HAS_PIL = _PIL_Image() is not None; HAS_STT = _speech_recognition() is not None
HAS_CLIP = _pyperclip() is not None; HAS_MATPLOTLIB = _matplotlib() is not None; HAS_CHROMADB = _chromadb() is not None
HAS_PORCUPINE = _pvporcupine() is not None; HAS_CV2 = _cv2() is not None
HAS_SYMPY = _sympy() is not None; HAS_NUMPY = _numpy() is not None

# ── DEPENDENCY CHECKER ──────────────────────────────────────
def check_dependencies(verbose=True):
    missing = []
    for name, present in [
        ("requests", True),
        ("chromadb", HAS_CHROMADB), ("pypdf", HAS_PYPDF),
        ("PIL", HAS_PIL), ("pygame", HAS_PYGAME),
        ("edge_tts", HAS_EDGE), ("pyttsx3", HAS_PYTTSX3),
        ("speech_recognition", HAS_STT), ("pyperclip", HAS_CLIP),
        ("matplotlib", HAS_MATPLOTLIB), ("pvporcupine", HAS_PORCUPINE),
        ("opencv-python", HAS_CV2), ("psutil", HAS_PSUTIL),
        ("sympy", HAS_SYMPY), ("numpy", HAS_NUMPY),
    ]:
        if not present:
            missing.append(name)
    if missing:
        msg = f"Optional deps missing: {', '.join(missing)}  ->  pip install {' '.join(missing)}"
        if verbose:
            print(f"⚠️ {msg}")
        logger.warning(msg)
    return missing

# ── Safe exec decorator ──────────────────────────────────
def safe_exec(func):
    def wrapper(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception as e:
            logger.error(f"[{func.__name__}] Error: {e}\n{traceback.format_exc()}")
            return f"❌ {e}"
    return wrapper

# ── Windows unicode fix ────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── DATABASE ──────────────────────────────────────────────────
_db_local = threading.local()
def get_db():
    db = getattr(_db_local, "conn", None)
    if db is None:
        db = sqlite3.connect(str(DB_FILE), check_same_thread=False)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, content TEXT, vec_json TEXT, project TEXT DEFAULT 'Default');
            CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT UNIQUE, ts REAL, project TEXT DEFAULT 'Default');
            CREATE TABLE IF NOT EXISTS agent_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, task TEXT, results TEXT, project TEXT DEFAULT 'Default');
            CREATE TABLE IF NOT EXISTS pc_actions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, action TEXT, details TEXT, result TEXT, approved INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS created_files(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, path TEXT, filetype TEXT, project TEXT DEFAULT 'Default');
            CREATE TABLE IF NOT EXISTS voice_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, transcript TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS embeddings_cache(text_hash TEXT PRIMARY KEY, embedding_json TEXT, created_ts REAL);
            CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, query TEXT, options TEXT, criteria TEXT, recommendation TEXT, confidence REAL, chosen TEXT, outcome TEXT);
        """)
        db.commit()
        _db_local.conn = db
    return db

# ── RESPONSE CACHE ────────────────────────────────────────────
_RESPONSE_CACHE: dict = {}
_cache_lock = threading.Lock()
def cache_key(model, system, messages):
    raw = json.dumps({"m": model, "s": system[:200], "msgs": messages[-3:]}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()
def cache_get(key):
    with _cache_lock:
        entry = _RESPONSE_CACHE.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["v"]
    return None
def cache_set(key, value):
    with _cache_lock:
        if len(_RESPONSE_CACHE) >= _CACHE_MAX:
            oldest = min(_RESPONSE_CACHE.items(), key=lambda x: x[1]["ts"])
            _RESPONSE_CACHE.pop(oldest[0], None)
        _RESPONSE_CACHE[key] = {"v": value, "ts": time.time()}

# ── EMBEDDING CACHE (in-memory + sqlite backed) ───────────────
_embed_cache_lock = threading.Lock()
_embed_memory_cache = {}
def _cache_embedding(text: str, embedding: list):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    with _embed_cache_lock:
        _embed_memory_cache[text_hash] = embedding
    try:
        db = get_db()
        db.execute("INSERT OR REPLACE INTO embeddings_cache(text_hash, embedding_json, created_ts) VALUES(?,?,?)",(text_hash, json.dumps(embedding), time.time()))
        db.commit()
    except Exception:
        pass
def _get_cached_embedding(text: str) -> Optional[list]:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    with _embed_cache_lock:
        if text_hash in _embed_memory_cache:
            return _embed_memory_cache[text_hash]
    try:
        row = get_db().execute("SELECT embedding_json FROM embeddings_cache WHERE text_hash=?",(text_hash,)).fetchone()
        if row:
            emb = json.loads(row[0])
            with _embed_cache_lock:
                _embed_memory_cache[text_hash] = emb
            return emb
    except Exception:
        pass
    return None

# ── OLLAMA EMBEDDING ─────────────────────────────────────────
@safe_exec
def get_embedding(text: str) -> list:
    cached = _get_cached_embedding(text)
    if cached:
        return cached
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/embeddings", json={"model": EMBEDDING_MODEL, "prompt": text}, timeout=30)
        if r.status_code == 200:
            embedding = r.json().get("embedding", [])
            if embedding:
                _cache_embedding(text, embedding)
            return embedding
        return []
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

# ── VECTOR STORE (ChromaDB, local persistent) ─────────────────
_vector_client = None; _vector_collection = None
_vector_lock = threading.Lock()
def get_vector_store():
    global _vector_client, _vector_collection
    if not HAS_CHROMADB:
        return None, None
    with _vector_lock:
        if _vector_client is None:
            try:
                chromadb = _chromadb()
                _vector_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH), settings=chromadb.config.Settings(anonymized_telemetry=False))
                _vector_collection = _vector_client.get_or_create_collection(name="omnix_memories", metadata={"hnsw:space": "cosine"})
            except Exception as e:
                logger.error(f"ChromaDB init error: {e}")
                return None, None
    return _vector_client, _vector_collection

@safe_exec
def vector_store_memory(content, role, project, memory_id=None):
    if not HAS_CHROMADB:
        return
    _, collection = get_vector_store()
    if collection is None:
        return
    embedding = get_embedding(content)
    if not embedding:
        return
    try:
        doc_id = str(memory_id) if memory_id else hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
        collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[{"role": role, "project": project, "timestamp": time.time()}], documents=[content])
    except Exception as e:
        logger.error(f"Vector store error: {e}")

@safe_exec
def vector_search(query, project, k=SEM_TOP_K):
    if not HAS_CHROMADB:
        return []
    _, collection = get_vector_store()
    if collection is None:
        return []
    embedding = get_embedding(query)
    if not embedding:
        return []
    try:
        results = collection.query(query_embeddings=[embedding], n_results=k, where={"project": project}, include=["documents","metadatas","distances"])
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist if dist else 0.0
            if score > 0.1:
                out.append({"content": doc, "role": meta.get("role","unknown"), "ts": meta.get("timestamp",0), "score": score})
        return out
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []

# ── TF-IDF MEMORY (offline fallback, no external deps) ────────
STOP = {"the","a","an","is","it","in","on","at","to","for","of","and","or","but","with","this","that","was","are","be","been","by","from","as","i","you","he","she","we","they","me","my","your","his","her","its"}
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]
def tfidf_vec(text, idf):
    tokens = tokenize(text)
    if not tokens:
        return {}
    tf = Counter(tokens); n = len(tokens)
    vec = {t: (c/n) * idf.get(t, log(MAX_MEM+1)) for t, c in tf.items()}
    norm = sqrt(sum(v*v for v in vec.values())) or 1.0
    return {k: v/norm for k, v in vec.items()}
def cos_sparse(a, b):
    c = set(a) & set(b)
    if not c:
        return 0.0
    dot = sum(a[k]*b[k] for k in c)
    na = sqrt(sum(v*v for v in a.values())) or 1.0
    nb = sqrt(sum(v*v for v in b.values())) or 1.0
    return dot/(na*nb)

_idf_cache = {"ts": 0.0, "value": {}}
_IDF_TTL = 60
def load_idf(force=False):
    if not force and (time.time() - _idf_cache["ts"] < _IDF_TTL):
        return _idf_cache["value"]
    try:
        rows = get_db().execute("SELECT content FROM memories").fetchall()
        if not rows:
            value = {}
        else:
            df = Counter(); N = len(rows)
            for (c,) in rows:
                df.update(set(tokenize(c)))
            value = {t: log(N/(f+1)) for t, f in df.items()}
    except Exception:
        value = {}
    _idf_cache["ts"] = time.time()
    _idf_cache["value"] = value
    return value
def invalidate_idf_cache():
    _idf_cache["ts"] = 0.0

def tfidf_search(query, project, k=SEM_TOP_K):
    try:
        idf = load_idf(); qv = tfidf_vec(query, idf)
        if not qv:
            return []
        rows = get_db().execute("SELECT ts,role,content,vec_json FROM memories WHERE project=? ORDER BY ts DESC LIMIT 400",(project,)).fetchall()
        scored = []
        for ts, role, content, vj in rows:
            try:
                mv = json.loads(vj or "{}")
            except Exception:
                mv = {}
            scored.append({"ts":ts,"role":role,"content":content,"score":cos_sparse(qv,mv)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [s for s in scored[:k] if s["score"] > 0.05]
    except Exception:
        return []

def sem_search(query, project, k=SEM_TOP_K):
    if HAS_CHROMADB:
        results = vector_search(query, project, k)
        if results:
            return results
    return tfidf_search(query, project, k)

def sem_context_block(query, project):
    mems = sem_search(query, project)
    if not mems:
        return ""
    lines = [f"[{datetime.fromtimestamp(m['ts']).strftime('%m/%d %H:%M')} {m['role']}] {m['content'][:180]}" for m in mems]
    return "=== SEMANTIC MEMORY ===\n" + "\n".join(lines) + "\n=== END ===\n\n"

@safe_exec
def store_memory(content, role, project):
    try:
        idf = load_idf(); vec = tfidf_vec(content, idf)
        db = get_db()
        cursor = db.execute("INSERT INTO memories(ts,role,content,vec_json,project) VALUES(?,?,?,?,?)",(time.time(), role, content, json.dumps(vec), project))
        memory_id = cursor.lastrowid; db.commit()
        invalidate_idf_cache()
        if HAS_CHROMADB:
            vector_store_memory(content, role, project, memory_id)
    except Exception as e:
        logger.error(f"store_memory error: {e}")

@safe_exec
def store_fact(fact, project):
    try:
        get_db().execute("INSERT OR IGNORE INTO facts(fact,ts,project) VALUES(?,?,?)",(fact, time.time(), project))
        get_db().commit()
    except Exception:
        pass

def load_facts(project):
    try:
        return [r[0] for r in get_db().execute("SELECT fact FROM facts WHERE project=? ORDER BY ts DESC LIMIT 50",(project,)).fetchall()]
    except Exception:
        return []

@safe_exec
def log_pc_action(action, details, result, approved):
    try:
        get_db().execute("INSERT INTO pc_actions(ts,action,details,result,approved) VALUES(?,?,?,?,?)",(time.time(), action, details[:500], result[:500], int(approved)))
        get_db().commit()
    except Exception:
        pass

def get_pc_history(limit=20):
    try:
        return get_db().execute("SELECT ts,action,details,result,approved FROM pc_actions ORDER BY ts DESC LIMIT ?",(limit,)).fetchall()
    except Exception:
        return []

@safe_exec
def log_created_file(path, filetype, project):
    try:
        get_db().execute("INSERT INTO created_files(ts,path,filetype,project) VALUES(?,?,?,?)",(time.time(), path, filetype, project))
        get_db().commit()
    except Exception:
        pass

@safe_exec
def log_voice(transcript, source):
    try:
        get_db().execute("INSERT INTO voice_log(ts,transcript,source) VALUES(?,?,?)",(time.time(), transcript[:500], source))
        get_db().commit()
    except Exception:
        pass

# ── KB ───────────────────────────────────────────────────────
def load_kb():
    if KB_FILE.exists():
        try:
            data = json.loads(KB_FILE.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []
def save_kb(kb_list):
    try:
        KB_FILE.write_text(json.dumps(kb_list, indent=2))
    except Exception:
        pass
def preload_critical_kb(state):
    current = list(state.kb) if getattr(state, "kb", None) else load_kb()
    state.kb = current
    save_kb(current)

# ── OLLAMA API (all localhost) ──
def _ollama_request_with_retry(func, *args, retries=None, backoff=None, **kwargs):
    retries = retries or OLLAMA_RETRIES
    backoff = backoff or OLLAMA_BACKOFF
    last_exc = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.RequestException) as e:
            last_exc = e
            wait = backoff ** i
            logger.warning(f"Ollama error: {e}. Retry {i+1}/{retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Ollama unreachable after {retries} retries: {last_exc}")

def ollama_status():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        if r.status_code == 200:
            return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return False, []

def _inject_system(msgs, system):
    if not system.strip():
        return msgs
    return [{"role": "system", "content": system.strip()}] + msgs

@safe_exec
def ollama_complete(msgs, model, temp=0.7, max_tok=1024, system="", use_cache=False):
    ck = None
    if use_cache or temp == 0.0:
        ck = cache_key(model, system, msgs)
        hit = cache_get(ck)
        if hit is not None:
            return hit
    def _call():
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json={"model": model, "messages": _inject_system(msgs, system), "stream": False, "options": {"temperature": temp, "num_predict": max_tok}}, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    result = _ollama_request_with_retry(_call)
    if ck:
        cache_set(ck, result)
    return result

class OllamaStreamError(Exception):
    pass

def ollama_stream(msgs, model, temp=0.7, max_tok=1024, system=""):
    def _call():
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json={"model": model, "messages": _inject_system(msgs, system), "stream": True, "options": {"temperature": temp, "num_predict": max_tok}}, stream=True, timeout=180)
        return r
    resp = _ollama_request_with_retry(_call)
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text or f"HTTP {resp.status_code}"
        raise OllamaStreamError(f"Ollama error ({resp.status_code}): {err}")
    got_any_token = False
    saw_done = False
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except Exception:
            continue
        if chunk.get("error"):
            raise OllamaStreamError(f"Ollama error: {chunk['error']}")
        tok = chunk.get("message", {}).get("content", "")
        if tok:
            got_any_token = True
            yield tok
        if chunk.get("done"):
            saw_done = True
            break
    if not got_any_token and not saw_done:
        raise OllamaStreamError(f"Model '{model}' returned no output. It may not be pulled — try `ollama pull {model}`.")

@safe_exec
def ollama_vision(image_b64, prompt, model):
    def _call():
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json={"model": model, "prompt": prompt, "images": [image_b64], "stream": False}, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    return _ollama_request_with_retry(_call)

# ── TTS ──────────────────────────────────────────────────────
EDGE_VOICES = {
    "Aria (US, Female)": "en-US-AriaNeural", "Jenny (US, Female)": "en-US-JennyNeural",
    "Guy (US, Male)": "en-US-GuyNeural", "Sonia (UK, Female)": "en-GB-SoniaNeural",
    "Ryan (UK, Male)": "en-GB-RyanNeural", "Natasha (AU, Female)": "en-AU-NatashaNeural",
    "Neerja (IN, Female)": "en-IN-NeerjaNeural", "Swara (IN, Hindi)": "hi-IN-SwaraNeural",
    "Madhur (IN, Hindi)": "hi-IN-MadhurNeural",
}
def clean_for_tts(text, max_chars=700):
    text = re.sub(r"```[\s\S]*?```", " (code block) ", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", "(link)", text)
    text = re.sub(r"[⚡🤖🧠🔍🔬📖🌿🎓💻🖥️📋✅❌⚠️📁📄🎙️🔊📌🌿↺🔐]", "", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars].rfind(".")
        text = text[:cut+1] if cut > max_chars // 2 else text[:max_chars]
    return text

async def _edge_tts_bytes(text, voice, rate="+0%"):
    edge_tts = _edge_tts()
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def edge_tts_speak(text, voice="en-US-AriaNeural", rate="+0%"):
    cleaned = clean_for_tts(text)
    if not cleaned.strip():
        return
    def _worker():
        if HAS_EDGE:
            try:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                audio = loop.run_until_complete(_edge_tts_bytes(cleaned, voice, rate))
                loop.close()
                if audio:
                    _play_mp3(audio)
                    return
            except Exception:
                pass
        if HAS_PYTTSX3:
            try:
                pyttsx3 = _pyttsx3(); engine = pyttsx3.init()
                engine.setProperty("rate", 170); engine.setProperty("volume", 0.9)
                engine.say(cleaned); engine.runAndWait(); engine.stop()
            except Exception:
                pass
    threading.Thread(target=_worker, daemon=True).start()

def offline_tts_speak(text, rate=170, volume=0.9):
    cleaned = clean_for_tts(text)
    if not cleaned.strip() or not HAS_PYTTSX3:
        return
    def _worker():
        try:
            pyttsx3 = _pyttsx3(); engine = pyttsx3.init()
            engine.setProperty("rate", rate); engine.setProperty("volume", volume)
            engine.say(cleaned); engine.runAndWait(); engine.stop()
        except Exception as e:
            logger.error(f"Offline TTS error: {e}")
    threading.Thread(target=_worker, daemon=True).start()

def _play_mp3(audio_bytes):
    if HAS_PYGAME:
        try:
            pygame = _pygame()
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
            pygame.mixer.music.load(io.BytesIO(audio_bytes), "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload(); return
        except Exception:
            pass
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        for player in [["ffplay","-nodisp","-autoexit"],["mpg123","-q"],["mplayer","-really-quiet"]]:
            try:
                subprocess.run(player+[tmp], timeout=60, capture_output=True); break
            except FileNotFoundError:
                continue
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def speak(text, state=None):
    voice = getattr(state, "tts_voice", "en-US-AriaNeural") if state else "en-US-AriaNeural"
    rate = getattr(state, "tts_rate_str", "+0%") if state else "+0%"
    if HAS_PYTTSX3:
        offline_tts_speak(text)
    else:
        threading.Thread(target=edge_tts_speak, args=(text, voice, rate), daemon=True).start()

# ── STT ──────────────────────────────────────────────────────
def voice_listen(timeout=6, phrase_limit=15):
    if not HAS_STT:
        return {"ok": False, "text": "", "error": "Install: pip install SpeechRecognition pyaudio"}
    sr = _speech_recognition(); r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.4)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        try:
            text = r.recognize_google(audio)
        except Exception:
            try:
                text = r.recognize_sphinx(audio)
            except Exception as e2:
                return {"ok": False, "text": "", "error": f"Recognition failed: {e2}"}
        log_voice(text, "mic")
        return {"ok": True, "text": text, "error": ""}
    except sr.WaitTimeoutError:
        return {"ok": False, "text": "", "error": "No speech detected."}
    except sr.UnknownValueError:
        return {"ok": False, "text": "", "error": "Couldn't understand."}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e)}

# ── WAKE WORD ENGINE ─────────────────────────────────────────
class WakeWordEngine:
    def __init__(self, callback=None, sensitivities=0.7, audio_device_index=None):
        self.callback = callback; self.sensitivities = sensitivities; self.audio_device_index = audio_device_index
        self._running = False; self._thread = None; self._porcupine = None; self._audio = None; self._stream = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        if not HAS_PORCUPINE:
            logger.error("Porcupine not available.")
            return False
        with self._lock:
            if self._running:
                return False
        try:
            pvporcupine = _pvporcupine()
            self._porcupine = pvporcupine.create(keywords=["porcupine","blueberry"], sensitivities=[self.sensitivities]*2)
            pyaudio = _pyaudio(); self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(rate=self._porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=self._porcupine.frame_length, input_device_index=self.audio_device_index)
            self._running = True; self._thread = threading.Thread(target=self._listen_loop, daemon=True); self._thread.start()
            logger.info("Wake word engine active — say 'Porcupine' or 'Blueberry'")
            return True
        except Exception as e:
            logger.error(f"Wake word init error: {e}")
            self._running = False; return False

    def _listen_loop(self):
        while self._running:
            try:
                pcm = self._stream.read(self._porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h"*self._porcupine.frame_length, pcm)
                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    self._on_wake_detected()
            except Exception as e:
                logger.error(f"Wake word loop error: {e}")
                time.sleep(0.5)

    def _on_wake_detected(self):
        try:
            if HAS_PYGAME:
                import numpy as np
                pygame = _pygame(); pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                t = np.linspace(0, 0.15, int(22050*0.15)); tone = np.sin(2*np.pi*880*t)*0.3
                sound = pygame.sndarray.make_sound((tone*32767).astype(np.int16)); sound.play()
        except Exception:
            pass
        if HAS_STT:
            sr = _speech_recognition(); r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.3)
                    audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio).strip()
                if text and self.callback:
                    self.callback(text)
            except Exception:
                pass

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop_stream(); self._stream.close()
        if self._audio:
            self._audio.terminate()
        if self._porcupine:
            self._porcupine.delete()
        logger.info("Wake word engine stopped.")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

# ── SANDBOX ──────────────────────────────────────────────────
_SAFE_BUILTINS = {"print":print,"range":range,"len":len,"enumerate":enumerate,"zip":zip,"map":map,"filter":filter,"sorted":sorted,"reversed":reversed,"list":list,"dict":dict,"set":set,"tuple":tuple,"int":int,"float":float,"str":str,"bool":bool,"abs":abs,"round":round,"min":min,"max":max,"sum":sum,"all":all,"any":any,"pow":pow,"divmod":divmod,"hex":hex,"oct":oct,"bin":bin,"isinstance":isinstance,"type":type,"repr":repr,"chr":chr,"ord":ord,"format":format,"iter":iter,"next":next,"hash":hash,"id":id,"callable":callable}
def _exec_worker(code, q):
    import contextlib
    safe = {"__builtins__": _SAFE_BUILTINS, "math": math}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(textwrap.dedent(code), safe)
        out = buf.getvalue().strip(); q.put(("ok", out if out else "(no output)"))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))
@safe_exec
def sandbox_exec(code):
    q = queue.Queue(); t = threading.Thread(target=_exec_worker, args=(code,q), daemon=True); t.start()
    t.join(timeout=SANDBOX_SEC)
    if t.is_alive():
        return f"❌ Killed — code exceeded {SANDBOX_SEC}s timeout"
    try:
        status, result = q.get_nowait()
        return (result[:4000]+"\n…(truncated)" if len(result)>4000 else result) if status=="ok" else f"❌ {result}"
    except queue.Empty:
        return "❌ No output received"

# ── AUTO-CODER ───────────────────────────────────────────────
class AutoCoder:
    @staticmethod
    def auto_fix(code, error, model="qwen2.5-coder:7b"):
        prompt = f"CODE:\n{code}\nERROR:\n{error}\nRewrite to fix. Output ONLY corrected code."
        try:
            return ollama_complete([{"role":"user","content":prompt}], model, temp=0.2, max_tok=800)
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
            return f"# Auto-fix failed: {e}\n{code}"
    @staticmethod
    def auto_test(code, model="llama3.2"):
        prompt = f"Write Python unit tests for this code. Output ONLY test code.\nCODE:\n{code}"
        try:
            return ollama_complete([{"role":"user","content":prompt}], model, temp=0.3, max_tok=600)
        except Exception as e:
            logger.error(f"Test gen failed: {e}")
            return f"# Test gen failed: {e}"
    @staticmethod
    def auto_refactor(code, model="qwen2.5-coder:7b"):
        prompt = f"Refactor this Python code cleaner. Output ONLY refactored.\nCODE:\n{code}"
        try:
            return ollama_complete([{"role":"user","content":prompt}], model, temp=0.2, max_tok=800)
        except Exception as e:
            logger.error(f"Refactor failed: {e}")
            return f"# Refactor failed: {e}\n{code}"
    @staticmethod
    def auto_explain(code, model="llama3.2"):
        prompt = f"Explain this code step by step.\nCODE:\n{code}"
        try:
            return ollama_complete([{"role":"user","content":prompt}], model, temp=0.3, max_tok=300)
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return f"# Explanation failed: {e}"
    @staticmethod
    def run_with_watchdog(code, timeout=10, retries=2):
        for attempt in range(retries+1):
            fpath = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code); fpath = f.name
                result = subprocess.run(['python', fpath], capture_output=True, text=True, timeout=timeout)
                os.unlink(fpath)
                if result.returncode == 0:
                    return {"success": True, "output": result.stdout, "attempts": attempt+1}
                else:
                    error = result.stderr or "Unknown error"
                    if attempt < retries:
                        code = AutoCoder.auto_fix(code, error)
                    else:
                        return {"success": False, "error": error, "attempts": attempt+1}
            except subprocess.TimeoutExpired:
                if fpath:
                    try:
                        os.unlink(fpath)
                    except Exception:
                        pass
                return {"success": False, "error": f"Timeout after {timeout}s", "attempts": attempt+1}
            except Exception as e:
                if attempt < retries:
                    code = AutoCoder.auto_fix(code, str(e))
                else:
                    return {"success": False, "error": str(e), "attempts": attempt+1}
        return {"success": False, "error": "Max retries exceeded", "attempts": retries+1}

# ── CALCULATOR ─────────────────────────────────
_OPS = {ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.Pow:op.pow,ast.USub:op.neg,ast.UAdd:op.pos,ast.Mod:op.mod,ast.FloorDiv:op.floordiv}
_NAMES = {k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
_NAMES.update({"abs":abs,"round":round,"int":int,"float":float})
def _aeval(n):
    if isinstance(n, ast.Expression):
        return _aeval(n.body)
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)):
            return n.value
        raise ValueError(f"Unsupported constant: {n.value!r}")
    if isinstance(n, ast.Name):
        if n.id in _NAMES:
            return _NAMES[n.id]
        raise ValueError(f"Unknown name: {n.id}")
    if isinstance(n, ast.BinOp):
        return _OPS[type(n.op)](_aeval(n.left),_aeval(n.right))
    if isinstance(n, ast.UnaryOp):
        return _OPS[type(n.op)](_aeval(n.operand))
    if isinstance(n, ast.Call):
        return _aeval(n.func)(*[_aeval(a) for a in n.args])
    raise ValueError(f"Unsupported expression: {ast.dump(n)}")
def tool_calc(expr):
    try:
        result = _aeval(ast.parse(expr.strip(), mode="eval"))
        return f"= {round(result,10) if isinstance(result,float) else result}"
    except Exception as e:
        return f"❌ Calc error: {e}"

@safe_exec
def tool_chart(data, chart_type="bar", title="", x_label="", y_label=""):
    if not HAS_MATPLOTLIB:
        return "❌ matplotlib not installed."
    try:
        matplotlib = _matplotlib(); matplotlib.use('Agg'); plt = _plt()
        fig, ax = plt.subplots(figsize=(8,4.5))
        x, y = data.get("x",[]), data.get("y",[])
        if not x or not y:
            return "❌ Missing data"
        if chart_type=="bar":
            ax.bar(x,y,color='#8ab4f8',edgecolor='#1a73e8',alpha=0.85)
        elif chart_type=="line":
            ax.plot(x,y,marker='o',linewidth=2.5,color='#8ab4f8',markersize=8)
        else:
            ax.scatter(x,y,s=80,color='#c58af9',alpha=0.8,edgecolors='#7c3aed')
        ax.set_title(title,fontsize=14,fontweight='bold'); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.grid(True,linestyle='--',alpha=0.3); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf,format='png',dpi=120,bbox_inches='tight'); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode(); plt.close(fig)
        return f"IMAGE:CHART:<img src='data:image/png;base64,{b64}' style='max-width:100%;border-radius:12px;margin:8px 0;'>"
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return f"❌ Chart error: {e}"

# ── TOOL ROUTER ────────────────────────────────────────────────
TOOL_ROUTER_PROMPT = """Decide if this message needs a tool. Chart/graph/plot -> chart. Music/song -> music. Calculate -> calculate. Current events -> search. Facts -> wikipedia. Code -> auto_code. Improve -> auto_improve. Else -> DIRECT. Output ONLY JSON or DIRECT."""
def extract_json(text):
    m = re.search(r"\{[^{}]*\}", re.sub(r"```(?:json)?","",text).replace("```",""), re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group())
        return o if "tool" in o else None
    except Exception:
        return None
@safe_exec
def route_tool(user_msg, model, temp):
    try:
        d = ollama_complete([{"role":"user","content":user_msg}], model=model, temp=0.0, max_tok=100, system=TOOL_ROUTER_PROMPT).strip()
        if d.upper().startswith("DIRECT"):
            return None
        return extract_json(d)
    except Exception:
        return None
@safe_exec
def dispatch_tool(call, model=""):
    t = call.get("tool","")
    if t=="calculate":
        return t, tool_calc(call.get("expr",""))
    if t=="search":
        return t, tool_search(call.get("query",""))
    if t=="python":
        return t, sandbox_exec(call.get("code",""))
    if t=="read_file":
        path = Path(call.get("path", "")).expanduser()
        if not path.is_file():
            return t, f"❌ File not found: {path}"
        try:
            return t, path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return t, f"❌ Could not read {path}: {e}"
    if t=="list_dir":
        path = Path(call.get("path", ".")).expanduser()
        if not path.is_dir():
            return t, f"❌ Directory not found: {path}"
        try:
            entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            return t, "\n".join(f"{'[DIR] ' if item.is_dir() else ''}{item.name}" for item in entries)
        except OSError as e:
            return t, f"❌ Could not list {path}: {e}"
    if t=="wikipedia":
        return t, tool_wikipedia(call.get("query",""))
    if t=="chart":
        return t, tool_chart(call.get("data",{}), call.get("type","bar"), call.get("title",""), call.get("x_label",""), call.get("y_label",""))
    if t=="auto_code":
        action = call.get("action","run"); code = call.get("code","")
        if action=="fix":
            result = AutoCoder.auto_fix(code, call.get("error",""))
        elif action=="test":
            result = AutoCoder.auto_test(code)
        elif action=="refactor":
            result = AutoCoder.auto_refactor(code)
        elif action=="explain":
            result = AutoCoder.auto_explain(code)
        else:
            result = AutoCoder.run_with_watchdog(code)
        return t, json.dumps(result, indent=2) if isinstance(result,dict) else result
    if t=="auto_improve":
        action = call.get("action","scan")
        if action=="scan":
            results = AutoImprover.full_improve()
        elif action=="apply":
            results = AutoImprover.full_improve(auto_apply=True)
        elif action=="log":
            return t, AutoImprover.get_log()
        else:
            return t, "❌ Unknown action"
        return t, json.dumps(results, indent=2)
    return t, f"❌ Unknown tool: {t}"

# ── AUTO-IMPROVER ────────────────────────────────────────────
class AutoImprover:
    IMPROVE_LOG = Path("omnix_improve_log.json")
    @staticmethod
    def scan_files(directory="."):
        return [p for p in Path(directory).rglob("*.py") if not any(x in p.parts for x in ["venv","__pycache__",".git"])]
    @staticmethod
    def analyze_file(filepath, model="qwen2.5-coder:7b"):
        content = filepath.read_text(encoding="utf-8")
        prompt = f"Optimize OmniX. Constraints: 16GB RAM, offline, Ollama. File: {filepath.name}\nContent: {content[:6000]}\nSuggest max 3 improvements as JSON: [{{\"suggestion\":..., \"original\":..., \"improved\":...}}]"
        try:
            resp = ollama_complete([{"role":"user","content":prompt}], model, temp=0.2, max_tok=800)
            match = re.search(r'\[.*\]', resp, re.S)
            return {"file":str(filepath), "suggestions":json.loads(match.group())} if match else {"file":str(filepath),"suggestions":[]}
        except Exception as e:
            logger.error(f"Analyze file error: {e}")
            return {"file":str(filepath),"error":str(e)}
    @staticmethod
    def test_code(filepath, timeout=5):
        try:
            r = subprocess.run(["python","-m","py_compile",str(filepath)], capture_output=True, text=True, timeout=timeout)
            return (True,"Syntax OK") if r.returncode==0 else (False,r.stderr or r.stdout)
        except Exception as e:
            return False,str(e)
    @staticmethod
    def apply_with_fallback(filepath, original, improved, log=True):
        backup = filepath.with_suffix(filepath.suffix+".bak")
        try:
            if "omnixcore" in filepath.name or "omnix_app" in filepath.name:
                subprocess.run(["git","add",str(filepath)], capture_output=True, timeout=5)
                subprocess.run(["git","commit","-m",f"Auto-backup: {filepath.name}"], capture_output=True, timeout=5)
            filepath.rename(backup); filepath.write_text(improved, encoding="utf-8")
            ok, msg = AutoImprover.test_code(filepath)
            if ok:
                backup.unlink()
                if log:
                    AutoImprover._log(filepath, original, improved, True)
                return {"success":True,"message":"Improvement applied","reverted":False}
            else:
                backup.rename(filepath)
                if log:
                    AutoImprover._log(filepath, original, improved, False, msg)
                return {"success":False,"message":f"Failed: {msg}","reverted":True}
        except Exception as e:
            if backup.exists():
                backup.rename(filepath)
            if log:
                AutoImprover._log(filepath, original, improved, False, str(e))
            return {"success":False,"message":str(e),"reverted":True}
    @staticmethod
    def _log(filepath, original, improved, success, error=""):
        entry = {"timestamp":time.time(),"file":str(filepath),"original":original[:500],"improved":improved[:500],"success":success,"error":error}
        try:
            data = json.loads(AutoImprover.IMPROVE_LOG.read_text()) if AutoImprover.IMPROVE_LOG.exists() else []
            data.append(entry); AutoImprover.IMPROVE_LOG.write_text(json.dumps(data[-100:], indent=2))
        except Exception:
            pass
    @staticmethod
    def full_improve(directory=".", model="qwen2.5-coder:7b", auto_apply=False):
        results = []
        for f in AutoImprover.scan_files(directory):
            analysis = AutoImprover.analyze_file(f, model)
            if analysis.get("suggestions"):
                for sug in analysis["suggestions"]:
                    orig, impr = sug.get("original",""), sug.get("improved","")
                    if orig and impr and orig != impr:
                        sug["applied"] = AutoImprover.apply_with_fallback(f, orig, impr) if auto_apply else None
            results.append(analysis)
        return results
    @staticmethod
    def get_log():
        if not AutoImprover.IMPROVE_LOG.exists():
            return "📋 No improvements logged yet."
        try:
            data = json.loads(AutoImprover.IMPROVE_LOG.read_text())
            lines = ["📋 **Improvement Log**", "="*40]
            for e in data[-10:]:
                status = "✅" if e["success"] else "❌"
                lines.append(f"{status} {Path(e['file']).name} — {e['timestamp']}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Log error: {e}"

# ── FILE CREATOR ─────────────────────────────────────────────
FILE_TEMPLATES = {
    ".py": "# {name}\n# Created by OmniX AI\n\ndef main():\n    pass\n\nif __name__=='__main__': main()\n",
    ".md": "# {name}\n\n> Created by OmniX AI — {date}\n\n## Overview\n\n\n",
    ".html": "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>{name}</title></head><body><h1>{name}</h1></body></html>\n",
    ".txt": "{name}\nCreated by OmniX AI — {date}\n\n",
}
@safe_exec
def create_file(path, content="", ai_generate=False, model="", project="Default"):
    p = Path(path.strip()).expanduser(); suffix = p.suffix.lower(); date_str = datetime.now().strftime("%Y-%m-%d")
    if not content:
        template = FILE_TEMPLATES.get(suffix, "# {name}\n# Created by OmniX AI\n")
        content = template.format(name=p.stem, date=date_str)
    try:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content, encoding="utf-8")
        log_created_file(str(p), suffix, project)
        return True, f"✅ Created: {p} ({len(content)} chars)", str(p)
    except Exception as e:
        logger.error(f"Create file error: {e}")
        return False, f"❌ {e}", ""

# ── MUSIC (stub) ──────────────────────────────────────────────
_MUSIC_LOCK = threading.Lock()
MUSIC_PLAYER = {"queue": [], "status": "idle", "current": None}
def music_play(query):
    with _MUSIC_LOCK:
        if query:
            MUSIC_PLAYER["queue"].append(query)
            MUSIC_PLAYER["status"] = "playing"
            MUSIC_PLAYER["current"] = query
        return f"🎵 Playing: {query or 'Queued'}"
def music_queue(query):
    with _MUSIC_LOCK:
        MUSIC_PLAYER["queue"].append(query)
        return f"📝 Queued: {query}"
def music_smart(mood, n=8):
    with _MUSIC_LOCK:
        return f"🎵 {mood} playlist ({n} tracks) — local library not configured."
def music_status():
    with _MUSIC_LOCK:
        return f"▶ {MUSIC_PLAYER['status']} — {MUSIC_PLAYER['current'] or 'idle'}"
def music_skip():
    with _MUSIC_LOCK:
        MUSIC_PLAYER["status"] = "skipped"
        return "⏭ Skipped"
def music_pause():
    with _MUSIC_LOCK:
        MUSIC_PLAYER["status"] = "paused"
        return "⏸ Paused"
def music_resume():
    with _MUSIC_LOCK:
        MUSIC_PLAYER["status"] = "playing"
        return "▶ Resumed"

@safe_exec
def analyze_image(image_data, prompt, model):
    if not image_data:
        return "❌ No image data"
    b64 = base64.b64encode(image_data).decode()
    try:
        return ollama_vision(b64, prompt, model)
    except Exception as e:
        logger.error(f"Vision error: {e}")
        return f"❌ Vision error: {e}"

def b64_to_image_html(b64, caption="", max_width="100%"):
    return f'<div style="text-align:center;margin:8px 0"><img src="data:image/png;base64,{b64}" style="max-width:{max_width};border-radius:10px">{"<div style=font-size:0.72rem;opacity:0.6>"+caption+"</div>" if caption else ""}</div>'

# ── WEB SEARCH (network — only used if tools_enabled and online) ──
@safe_exec
def tool_search(query):
    results = []
    try:
        BeautifulSoup = _BeautifulSoup()
        r = requests.get("https://www.bing.com/search", params={"q":query,"setlang":"en"}, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
        if r.status_code == 200 and BeautifulSoup:
            soup = BeautifulSoup(r.text,"html.parser")
            for item in soup.select("li.b_algo")[:5]:
                t = (item.select_one("h2 a") or item).get_text(strip=True)
                s = (item.select_one(".b_caption p") or item).get_text(strip=True)
                if t:
                    results.append(f"**{t}**\n{s}"[:280])
    except Exception:
        pass
    try:
        s = requests.get("https://en.wikipedia.org/w/api.php", params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":3}, timeout=6, headers={"User-Agent":"OmniX/9.1"})
        wiki = s.json().get("query",{}).get("search",[])
        if wiki:
            title = wiki[0]["title"]
            r2 = requests.get("https://en.wikipedia.org/w/api.php", params={"action":"query","titles":title,"prop":"extracts","exintro":True,"explaintext":True,"format":"json","exsentences":5}, timeout=6)
            pages = r2.json().get("query",{}).get("pages",{}); page = next(iter(pages.values()))
            extract = page.get("extract","").strip()
            if extract:
                results.append(f"**Wikipedia: {title}**\n{extract[:600]}")
    except Exception:
        pass
    return "\n\n".join(results) if results else "⚠️ Search unavailable (offline, or no network access)."

@safe_exec
def tool_wikipedia(query):
    try:
        s = requests.get("https://en.wikipedia.org/w/api.php", params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":3}, timeout=6)
        results = s.json().get("query",{}).get("search",[])
        if not results:
            return f"No article for: {query}"
        title = results[0]["title"]
        r = requests.get("https://en.wikipedia.org/w/api.php", params={"action":"query","titles":title,"prop":"extracts","exintro":True,"explaintext":True,"format":"json","exsentences":8}, timeout=6)
        pages = r.json().get("query",{}).get("pages",{}); page = next(iter(pages.values()))
        extract = page.get("extract","").strip()
        return f"**{title}**\n{extract[:2500]}"
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return f"❌ Wikipedia unavailable — are you online? ({e})"

# ── TOOL ROUTER (agents) ────────────────────────────────────────
def load_custom_agents():
    if AGENTS_FILE.exists():
        try:
            return json.loads(AGENTS_FILE.read_text())
        except Exception:
            pass
    return {}
def save_custom_agents(ag):
    AGENTS_FILE.write_text(json.dumps(ag, indent=2))
def all_agents():
    return {**BUILTIN_AGENTS, **load_custom_agents()}
BUILTIN_AGENTS = {
    "planner": "Break complex tasks into clear numbered sub-tasks. Output a numbered plan only.",
    "researcher": "Gather relevant facts, considerations, and knowledge on the topic. Be thorough.",
    "coder": "Write clean, working, well-commented code with usage examples and edge case handling.",
    "critic": "Review for (1) errors/bugs (2) missing edge cases (3) improvements. Be specific.",
    "synthesizer":"Given all agent outputs, produce one coherent, high-quality final answer.",
    "debugger": "Analyze issues, identify root causes, provide step-by-step fixes.",
    "explainer": "Take complex content and explain it clearly with analogies for non-experts.",
    "tester": "Generate comprehensive test cases, edge cases, and validation strategies.",
    "dj": "You are the DJ agent. You recommend songs, build playlists, and manage music queues.",
}
@safe_exec
def run_agent(name, task, model, temp=0.7):
    system = all_agents().get(name, f"You are the {name} agent.")
    try:
        return ollama_complete([{"role":"user","content":task}], model=model, temp=temp, max_tok=1500, system=system)
    except Exception as e:
        logger.error(f"Run agent {name} error: {e}")
        return f"[{name} failed: {e}]"
def run_chain(chain, task, model):
    results = {"task":task,"chain":chain,"steps":{}}
    ctx = task
    for i,name in enumerate(chain):
        inp = f"Original: {task}\nContext:\n{ctx}" if i>0 else task
        out = run_agent(name, inp, model)
        results["steps"][name] = out; ctx += f"\n\n### {name.upper()}:\n{out}"
    results["final"] = results["steps"].get(chain[-1],"")
    return results

# ── PLANNING AGENT LOOP ─────────────────────────────────────────
def _plan_steps(goal, model, max_steps=6):
    prompt = f"""Break this goal into at most {max_steps} concrete, ordered sub-tasks.
Each sub-task should be small enough to execute and verify on its own.
Goal: {goal}

Respond ONLY with a JSON list of strings, e.g.:
["step one description", "step two description"]"""
    raw = ollama_complete(
        [{"role": "user", "content": prompt}], model=model, temp=0.2, max_tok=500,
        system="You are a precise planning agent. Output only valid JSON, nothing else.",
    )
    try:
        match = re.search(r"\[.*\]", raw, re.S)
        steps = json.loads(match.group(0)) if match else []
        steps = [s for s in steps if isinstance(s, str) and s.strip()]
        return steps[:max_steps] if steps else [goal]
    except Exception as e:
        logger.error(f"Plan parse error: {e}")
        return [goal]

def _execute_step(step, ctx, model):
    needs_compute = any(k in step.lower() for k in
                         ["calculate", "compute", "sum", "count", "sort", "how many"])
    if needs_compute:
        code_prompt = f"""Write Python code to accomplish this sub-task, using print()
for the final result. Output ONLY the code, no explanation.
Sub-task: {step}
Context so far:
{ctx[-2000:]}"""
        code = ollama_complete(
            [{"role": "user", "content": code_prompt}], model=model, temp=0.1, max_tok=400,
            system="You write short, correct, self-contained Python. Output only code.",
        )
        code = re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()
        result = sandbox_exec(code)
        if not str(result).startswith("❌"):
            return f"[computed via sandbox]\n{result}"
    prompt = f"""Sub-task: {step}
Context so far:
{ctx[-2000:]}

Complete this sub-task directly. Be concrete and specific."""
    return ollama_complete(
        [{"role": "user", "content": prompt}], model=model, temp=0.4, max_tok=700,
        system="You execute one specific sub-task precisely. Do not restate the whole plan.",
    )

def _check_step(step, result, model):
    prompt = f"""Sub-task: {step}
Result produced: {result[:1500]}

Does this result actually and fully satisfy the sub-task? Answer with exactly
one line starting with "YES" or "NO", followed by a one-sentence reason."""
    verdict = ollama_complete(
        [{"role": "user", "content": prompt}], model=model, temp=0.0, max_tok=100,
        system="You are a strict, skeptical reviewer. Default to NO if uncertain.",
    ).strip()
    ok = verdict.upper().startswith("YES")
    return ok, verdict

def run_plan(goal, model, max_steps=6, max_replans=2, on_progress=None):
    def _p(stage, detail=""):
        if on_progress:
            try:
                on_progress(stage, detail)
            except Exception:
                pass

    steps = _plan_steps(goal, model, max_steps)
    _p("plan", f"{len(steps)} steps: {steps}")

    ctx = f"Goal: {goal}\n"
    results, checks = [], []
    replans_used = 0
    i = 0
    while i < len(steps):
        step = steps[i]
        _p("executing", step)
        result = _execute_step(step, ctx, model)
        ok, note = _check_step(step, result, model)
        _p("checked", f"{'✅' if ok else '⚠️'} {note}")

        results.append(result)
        checks.append({"step": step, "ok": ok, "note": note})
        ctx += f"\n\n### Step: {step}\nResult: {result}\nCheck: {note}\n"

        if not ok and replans_used < max_replans:
            replans_used += 1
            _p("replanning", f"step {i+1} failed its check — revising remaining plan")
            remaining_goal = f"""Original goal: {goal}
Progress so far:
{ctx[-3000:]}

The previous step did not fully satisfy its sub-task ({note}).
Give a revised ordered list of remaining sub-tasks (at most {max_steps - i}) to
actually reach the original goal from here. Respond ONLY with a JSON list."""
            new_remaining = _plan_steps(remaining_goal, model, max_steps - i)
            steps = steps[:i + 1] + new_remaining
        i += 1

    synth_prompt = f"""Goal: {goal}
Here is everything produced across all sub-tasks:
{ctx[-4000:]}

Write the final, complete answer to the original goal, synthesizing the
sub-task results into one coherent response. Do not just list the steps —
actually answer the goal."""
    final = ollama_complete(
        [{"role": "user", "content": synth_prompt}], model=model, temp=0.4, max_tok=1200,
        system="You synthesize multi-step work into one clear final answer.",
    )
    _p("done", "")

    return {
        "goal": goal, "steps": steps, "results": results,
        "checks": checks, "replans_used": replans_used, "final": final,
    }

# ── EXPORT ────────────────────────────────────────────────────
def export_md(msgs, project="Chat"):
    lines = [f"# {project}\n\n*Exported {datetime.now():%Y-%m-%d %H:%M}*\n\n---\n\n"]
    for m in msgs:
        role = "**You**" if m["role"]=="user" else "**OmniX**"
        lines.append(f"### {role}  `{m.get('ts','')}`\n\n{m['content']}\n\n---\n\n")
    return "".join(lines)
def export_pdf(msgs, project="Chat"):
    if not HAS_FPDF:
        return None
    pdf = _fpdf()(); pdf.add_page()
    pdf.set_font("Helvetica","B",16); pdf.cell(0,10,project[:60],ln=True)
    pdf.set_font("Helvetica","",8); pdf.cell(0,6,datetime.now().strftime("%Y-%m-%d %H:%M"),ln=True); pdf.ln(4)
    for m in msgs:
        role = "You" if m["role"]=="user" else "OmniX"
        pdf.set_font("Helvetica","B",10); pdf.cell(0,7,f"{role}:",ln=True)
        pdf.set_font("Helvetica","",9)
        for line in m["content"][:2000].split("\n"):
            try:
                pdf.multi_cell(0,5,line[:180])
            except Exception:
                pass
        pdf.ln(3)
    return bytes(pdf.output())
def ingest_doc(file_path):
    try:
        p = Path(file_path)
        suf = p.suffix.lower()
        if suf==".pdf" and HAS_PYPDF:
            with open(p, "rb") as f:
                return "\n".join(pg.extract_text() or "" for pg in _pypdf().PdfReader(f).pages)
        if suf==".docx" and HAS_DOCX:
            return "\n".join(par.text for par in _docx().Document(str(p)).paragraphs)
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Ingest doc error: {e}")
        return f"[Error: {e}]"

# ── THINKING ──────────────────────────────────────────────────
THINKING_SYSTEM = """You are OmniX. Use <think> for deep reasoning then give final answer."""
def split_think_answer(full):
    m = re.search(r'<think>(.*?)</think>', full, re.S|re.I)
    if m:
        thinking = m.group(1).strip(); after = full[m.end():].strip()
        return thinking, re.sub(r'</think>','',after,flags=re.I).strip()
    return "", full.strip()
def count_think_words(thinking):
    words = len(thinking.split()); secs = max(1,words//40)
    if secs<5: return "Thought a moment"
    if secs<15: return f"Thought {secs}s"
    return f"Deep thought ({words} words)"
def stream_with_thinking(msgs, model, temp=0.6, max_tok=2048, system=THINKING_SYSTEM):
    full = ""
    for tok in ollama_stream(msgs, model, temp, max_tok, system):
        full += tok
    return split_think_answer(full)

# ── MEMORY COMPRESSION ────────────────────────────────────────
def auto_summarize_memory(messages, project, model="llama3.2", max_old=20):
    if len(messages) < 8:
        return ""
    old = messages[:max_old] if len(messages) > max_old else messages[:-5]
    if not old:
        return ""
    text = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in old)
    prompt = f"Summarize: 3-5 facts, 2-3 decisions, 1 sentence summary.\n{text}"
    try:
        summary = ollama_complete([{"role":"user","content":prompt}], model, temp=0.3, max_tok=400, use_cache=True)
        store_memory(f"[AUTO-SUMMARY] {summary}", "system", project)
        for line in summary.split("\n"):
            if line.strip().startswith("- "):
                store_fact(line.strip()[2:], project)
        return summary
    except Exception:
        return ""
def maybe_auto_compress(state):
    if len(state.messages) % CFG["auto_compress_threshold"] == 0 and len(state.messages) > 15:
        auto_summarize_memory(state.messages, state.active_project)

# ── SELF REPORT ───────────────────────────────────────────────
def self_report(state=None):
    connected, models = ollama_status()
    return f"""🩺 **OmniX v9.1** | Local offline assistant
Ollama: {'✅' if connected else '❌'} | Models: {', '.join(models) if models else 'None'}
Memory: ChromaDB {'✅' if HAS_CHROMADB else '❌ (TF-IDF fallback active)'} | TTS: {'✅ offline (pyttsx3)' if HAS_PYTTSX3 else ('✅ online (edge-tts)' if HAS_EDGE else '❌')} | Wake: {'✅' if HAS_PORCUPINE else '❌'}
Ready."""

# ── DEEP RESEARCH (multi-source synthesis) ────────────────────
def deep_research(query: str, model: str = "llama3.2") -> str:
    """Perform a multi-query web research and synthesize findings.
    This is the 'Deep Research' feature. It collects multiple sources
    and returns a structured summary."""
    if not query:
        return ""

    # Generate sub-queries to broaden the research
    sub_queries = [query]
    try:
        prompt = f"""Given the research question "{query}", generate 3 related but distinct search queries.
Output each query on a new line, no numbering."""
        resp = ollama_complete(
            [{"role": "user", "content": prompt}], model=model, temp=0.3, max_tok=150,
            system="You generate search queries. Output only queries, one per line.",
        )
        for line in resp.strip().split("\n"):
            line = line.strip()
            if line and line not in sub_queries:
                sub_queries.append(line)
    except Exception:
        pass

    # Gather search results for each sub-query
    all_results = []
    for sq in sub_queries[:4]:  # limit to 4 queries to stay fast
        try:
            res = tool_search(sq)
            if res and not res.startswith("⚠️"):
                all_results.append(f"**Search query:** {sq}\n{res}")
        except Exception as e:
            logger.error(f"Deep research search error for '{sq}': {e}")

    if not all_results:
        return "⚠️ Deep research could not gather any information (offline or search failed)."

    # Synthesize into a research brief
    combined = "\n\n".join(all_results)
    synth_prompt = f"""You are a research assistant. Based on the following search results, produce a concise research brief on:
"{query}"

Search results:
{combined[:6000]}

Structure:
1. Key findings (bullet points)
2. Main facts with source hints
3. Any contradictions or uncertainties
4. A one-paragraph conclusion

Use Markdown formatting. Keep it under 500 words."""
    try:
        brief = ollama_complete(
            [{"role": "user", "content": synth_prompt}], model=model, temp=0.4, max_tok=800,
            system="You synthesize multiple sources into a clear research brief.",
        )
        return f"📚 **Deep Research Brief**\n\n{brief}"
    except Exception as e:
        logger.error(f"Deep research synthesis error: {e}")
        return f"❌ Deep research failed during synthesis: {e}"

# ── VOICE CONVERSATION ───────────────────────────────────────
def voice_conversation(user_text, model, state, system=""):
    use_thinking = getattr(state, "thinking_mode", False)
    if use_thinking:
        thinking, final = stream_with_thinking([{"role":"user","content":user_text}], model, temp=0.7, max_tok=600, system=system or THINKING_SYSTEM)
        if not final:
            final = thinking
    else:
        final = ollama_complete([{"role":"user","content":user_text}], model, temp=0.7, max_tok=600, system=system)
    speak(final, state); log_voice(user_text,"mic")
    return {"reply":final}

# ── TELEMETRY ─────────────────────────────────────────────────
def get_telemetry(project="Default", total_sent=0, response_times=None):
    rt = response_times or []
    return {"timestamp":datetime.now().isoformat(), "project":project, "messages":total_sent, "avg_response":sum(rt)/max(len(rt),1)}
def log_telemetry(project="Default", total_sent=0, response_times=None):
    try:
        data = json.loads(TELEMETRY_FILE.read_text()) if TELEMETRY_FILE.exists() else []
        data.append(get_telemetry(project, total_sent, response_times or []))
        TELEMETRY_FILE.write_text(json.dumps(data[-1000:], indent=2))
    except Exception:
        pass
def get_telemetry_dashboard():
    if not TELEMETRY_FILE.exists():
        return "No data yet"
    data = json.loads(TELEMETRY_FILE.read_text())
    if not data:
        return "No data"
    latest = data[-1]
    return f"📊 **Telemetry** | Messages: {latest.get('messages',0)} | Avg Response: {latest.get('avg_response',0):.2f}s | Logs: {len(data)}"

# ── TASK ENGINE ───────────────────────────────────────────────
class TaskEngine:
    @staticmethod
    def add_task(task_type, payload, priority=0):
        tasks = TaskEngine._load(); tid = hashlib.md5(f"{task_type}{time.time()}".encode()).hexdigest()[:8]
        tasks[tid] = {"id":tid,"type":task_type,"payload":payload,"status":"pending","priority":priority,"created":time.time(),"started":None,"completed":None,"result":None}
        TaskEngine._save(tasks); return tid
    @staticmethod
    def list_tasks(status=None):
        tasks = TaskEngine._load()
        return [t for t in tasks.values() if not status or t["status"]==status]
    @staticmethod
    def _load():
        if TASK_FILE.exists():
            try:
                return json.loads(TASK_FILE.read_text())
            except Exception:
                return {}
        return {}
    @staticmethod
    def _save(tasks):
        TASK_FILE.write_text(json.dumps(tasks, indent=2, default=str))

# ── WORLD STATE ───────────────────────────────────────────────
class WorldState:
    @staticmethod
    def load():
        if WORLD_FILE.exists():
            try:
                return json.loads(WORLD_FILE.read_text())
            except Exception:
                return {}
        return {}
    @staticmethod
    def save(state):
        WORLD_FILE.write_text(json.dumps(state, indent=2, default=str))
    @staticmethod
    def update_project(name, updates):
        state = WorldState.load(); state.setdefault("projects",{}).setdefault(name,{}).update(updates); WorldState.save(state)
    @staticmethod
    def add_goal(goal):
        state = WorldState.load()
        if goal not in state.get("goals",[]):
            state.setdefault("goals",[]).append(goal); WorldState.save(state)
    @staticmethod
    def get_summary():
        state = WorldState.load()
        lines = ["🌍 **World State**"]
        for name, data in state.get("projects",{}).items():
            lines.append(f"- {name}: {data.get('status','?')}")
        lines.append(f"\n**Goals:** {', '.join(state.get('goals',['None']))}")
        return "\n".join(lines)

# ── KNOWLEDGE GRAPH ───────────────────────────────────────────
class KnowledgeGraph:
    @staticmethod
    def _load():
        if GRAPH_FILE.exists():
            try:
                return json.loads(GRAPH_FILE.read_text())
            except Exception:
                return {"nodes":{},"edges":[]}
        return {"nodes":{},"edges":[]}
    @staticmethod
    def _save(graph):
        GRAPH_FILE.write_text(json.dumps(graph, indent=2, default=str))
    @staticmethod
    def add_edge(e1, rel, e2):
        graph = KnowledgeGraph._load()
        if [e1,rel,e2] not in graph["edges"]:
            graph["edges"].append([e1,rel,e2])
        KnowledgeGraph._save(graph)
    @staticmethod
    def get_summary():
        graph = KnowledgeGraph._load()
        lines = ["🕸️ **Knowledge Graph**", f"Edges: {len(graph['edges'])}"]
        for e1,rel,e2 in graph["edges"][:20]:
            lines.append(f"- {e1} **{rel}** {e2}")
        return "\n".join(lines)
    @staticmethod
    def query(entity: str) -> list:
        graph = KnowledgeGraph._load()
        results = []
        for e1, rel, e2 in graph["edges"]:
            if e1.lower() == entity.lower():
                results.append((rel, e2))
            elif e2.lower() == entity.lower():
                results.append((rel, e1))
        return results

# ── REFLECTION ────────────────────────────────────────────────
def tool_reflect(messages, model="llama3.2"):
    if len(messages) < 3:
        return "Not enough to reflect."
    prompt = f"Rate your answers (1-10) and suggest 3 improvements.\n{json.dumps(messages[-10:])}"
    try:
        return f"🧠 **Self-Reflection**\n{ollama_complete([{'role':'user','content':prompt}], model, temp=0.3, max_tok=600)}"
    except Exception:
        return "❌ Reflection error"
def tool_summarize_chat(messages, model="llama3.2"):
    if not messages:
        return "No messages."
    text = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in messages[-20:])
    return ollama_complete([{"role":"user","content":f"Summarize:\n{text}"}], model, temp=0.3, max_tok=200)

# ── SYMBOLIC VERIFIER ──────────────────────────────────────
def symbolic_verify(query: str, project: str) -> Optional[str]:
    ql = query.lower().strip()
    facts = load_facts(project)
    for fact in facts:
        if ql in fact.lower() or fact.lower() in ql:
            return f"📌 **From memory:** {fact}"
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    for entity in entities:
        relations = KnowledgeGraph.query(entity)
        if relations:
            lines = [f"🕸️ **Knowledge Graph — {entity}:**"]
            for rel, target in relations[:5]:
                lines.append(f"- {entity} **{rel}** {target}")
            return "\n".join(lines)
    recent = sem_search(query, project, k=3)
    if recent and recent[0].get("score",0) > 0.8:
        return f"🧬 **From memory:** {recent[0]['content'][:500]}"
    return None

# ── WATCHDOG ──────────────────────────────────────────────────
_WATCHDOG_START = time.time()
def _wd_log(msg, level="INFO"):
    try:
        with open(WATCHDOG_LOG,"a") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}\n")
    except Exception:
        pass
def watchdog_status():
    ok, models = ollama_status()
    return {"ollama_healthy":ok, "model":models[0] if models else "N/A", "uptime":f"{int((time.time()-_WATCHDOG_START)//60)}m", "platform":"Online" if ok else "Degraded"}
def watchdog_restart_ollama():
    try:
        r = subprocess.run(["systemctl","restart","ollama"], timeout=10, capture_output=True)
        if r.returncode == 0:
            return "✅ Restarted via systemctl"
    except Exception:
        pass
    try:
        subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        ok, _ = ollama_status()
        return "✅ Restarted (ollama serve)" if ok else "⚠️ Launched but not responding yet"
    except Exception as e:
        return f"❌ Failed: {e}"
def get_watchdog_report():
    s = watchdog_status()
    return f"🛡️ **Watchdog** | Ollama: {'✅' if s['ollama_healthy'] else '❌'} | Uptime: {s['uptime']} | Model: {s['model']}"
def _watchdog_worker():
    while True:
        try:
            if not watchdog_status()["ollama_healthy"]:
                logger.warning("Watchdog: Ollama unhealthy. Restarting...")
                watchdog_restart_ollama()
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
        time.sleep(WATCHDOG_INTERVAL)
threading.Thread(target=_watchdog_worker, daemon=True).start()

# ── DREAM WORKER ─────────────────────────────────────────────
def dream_worker(state):
    if not getattr(state, "dreams_enabled", True):
        return
    try:
        yesterday = time.time() - 86400
        db = get_db()
        rows = db.execute("SELECT role, content FROM memories WHERE ts > ? ORDER BY ts DESC LIMIT 200", (yesterday,)).fetchall()
        if len(rows) < 10:
            return
        combined = "\n".join(f"[{r[0]}] {r[1][:500]}" for r in rows)
        dream_prompt = f"""You are OmniX's reflective process. Analyze today's conversation fragments and identify:
1. 3-5 emerging themes
2. 2-3 decisions that were being weighed
3. Any notable shifts in tone or focus
4. One actionable suggestion based on today's patterns

Conversation fragments:
{combined[:8000]}
Be concise. Output as bullet points."""
        dream_result = ollama_complete([{"role":"user","content":dream_prompt}], model="llama3.2", temp=0.4, max_tok=500)
        store_memory(f"[REFLECTION {datetime.now().strftime('%Y-%m-%d')}] {dream_result}", "system", state.active_project)
        for line in dream_result.split("\n"):
            line = line.strip()
            if line.startswith("- ") and len(line) > 10:
                store_fact(f"Insight: {line[2:]}", state.active_project)
        logger.info("Reflection complete — patterns extracted and stored.")
    except Exception as e:
        logger.error(f"Dream error: {e}")

# ── NIGHTLY IMPROVE ──────────────────────────────────────────
def nightly_improve_worker(state):
    if not getattr(state, "auto_improve_enabled", True):
        return
    try:
        results = AutoImprover.full_improve(directory=".", auto_apply=False)
        total_suggestions = 0
        report_lines = [f"[AUTO-IMPROVE REPORT {datetime.now().strftime('%Y-%m-%d')}]"]
        for r in results:
            file = Path(r["file"]).name
            if "error" in r:
                report_lines.append(f"❌ {file}: {r['error']}")
            elif r.get("suggestions"):
                count = len(r["suggestions"]); total_suggestions += count
                report_lines.append(f"📄 {file}: {count} suggestion(s)")
                for sug in r["suggestions"]:
                    report_lines.append(f"  - {sug['suggestion'][:100]}")
        report = "\n".join(report_lines)
        store_memory(report, "system", state.active_project)
        with open("omnix_improve_nightly.log","a") as f:
            f.write(report + "\n---\n")
        logger.info(f"Nightly auto-improve complete — {total_suggestions} suggestions found.")
    except Exception as e:
        logger.error(f"Nightly improve error: {e}")

# ── MULTI-MODAL (video/audio) ─────────────────────────────────
def process_video(file_path: str, model: str = "llava:latest") -> str:
    try:
        cv2 = _cv2()
        if not cv2:
            return "❌ OpenCV not installed."
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return "❌ Cannot open video file."
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS); duration = total_frames/fps if fps>0 else 0
        frames = []
        for i in range(5):
            frame_idx = int(i*total_frames/5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                b64 = base64.b64encode(buffer).decode()
                frames.append(b64)
        cap.release()
        if not frames:
            return "❌ Could not extract frames."
        descriptions = []
        for i, frame_b64 in enumerate(frames):
            desc = ollama_vision(frame_b64, "Describe what you see in this frame from a video. Be concise.", model)
            descriptions.append(f"Frame {i+1}: {desc[:300]}")
        summary_prompt = f"""A video was analyzed. Here are descriptions of 5 frames:
{chr(10).join(descriptions)}
Video duration: {duration:.1f}s, {total_frames} frames.
Summarize what happens in this video in 3-5 sentences."""
        summary = ollama_complete([{"role":"user","content":summary_prompt}], model="llama3.2", temp=0.3, max_tok=300)
        return f"""🎬 **Video Analysis** ({duration:.1f}s)
**Summary:** {summary}
**Frame Descriptions:**
{chr(10).join(descriptions)}"""
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return f"❌ Video processing error: {e}"

def process_audio(file_path: str) -> str:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return f"🎙️ **Audio Transcript (offline/Whisper):**\n{result['text']}"
    except ImportError:
        try:
            sr = _speech_recognition(); r = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            return f"🎙️ **Audio Transcript (online fallback):**\n{text}"
        except Exception as e:
            logger.error(f"Audio processing error (fallback): {e}")
            return "❌ Whisper not installed. Run: pip install openai-whisper (for offline transcription)"
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return f"❌ Audio processing error: {e}"

# ── SCHEDULER ────────────────────────────────────────────────
class AutoScheduler:
    _tasks = []; _running = False
    @classmethod
    def start(cls, state):
        if not cls._running:
            cls._running = True
            threading.Thread(target=cls._run, daemon=True).start()
            cls.add_task("memory_dream", lambda: dream_worker(state), interval=86400)
            cls.add_task("nightly_improve", lambda: nightly_improve_worker(state), interval=86400)
    @classmethod
    def _run(cls):
        while cls._running:
            now = time.time()
            for t in cls._tasks:
                if now - t.get("last_run",0) >= t.get("interval",3600):
                    try:
                        t["func"](); t["last_run"] = now
                    except Exception as e:
                        logger.error(f"Scheduler task {t['name']} error: {e}")
            time.sleep(30)
    @classmethod
    def add_task(cls, name, func, interval=3600):
        cls._tasks.append({"name":name,"func":func,"interval":interval,"last_run":0})

# ── BUILD SYSTEM PROMPT (unified, no modes) ──────────────────
CORE_BEHAVIOR_RULES = """=== CORE BEHAVIOR RULES ===
1. Be accurate over agreeable. If the user's premise is wrong, say so — do not
   quietly go along with an incorrect assumption.
2. If you don't know something or the provided context doesn't cover it, say
   "I don't know" or "I'm not certain" rather than inventing an answer.
3. Prefer concrete, specific answers over vague generalities. Give real
   numbers, real code, real steps — not filler.
4. Match response length to the question. Simple questions get short
   answers. Complex/technical questions get thorough ones. Don't pad.
5. When giving code: it must be complete and runnable, not pseudocode
   pretending to be code, unless pseudocode was explicitly requested.
6. Never fabricate tool results, file contents, or facts about the user's
   system that you have not actually retrieved.
=== END CORE BEHAVIOR RULES ===
"""

def build_sys(thinking, tools, project, query, state):
    parts = []

    # Identity
    parts.append(
        "=== IDENTITY ===\n"
        "You are OmniX, a local, offline-first AI assistant.\n"
        "=== END IDENTITY ===\n"
    )

    # Grounding
    grounding_parts = []
    if query:
        sym = symbolic_verify(query, project)
        if sym:
            grounding_parts.append(f"[Verified fact]\n{sym}")
    if getattr(state, "use_sem_mem", True) and query:
        sem = sem_context_block(query, project)
        if sem:
            grounding_parts.append(sem.strip())
    facts = load_facts(project)
    if facts:
        grounding_parts.append("[Stored facts]\n" + "\n".join(f"- {f}" for f in facts[-25:]))
    kb = getattr(state, "kb", [])
    if kb:
        kb_text = "\n\n".join(f"[Doc {i+1}]:\n{d[:3000]}" for i, d in enumerate(kb[:4]))
        grounding_parts.append(f"[Knowledge base excerpts]\n{kb_text}")
    if grounding_parts:
        parts.append(
            "=== GROUNDING CONTEXT (treat as authoritative for this turn; "
            "if it conflicts with your general knowledge, trust this) ===\n"
            + "\n\n".join(grounding_parts) +
            "\n=== END GROUNDING CONTEXT ==="
        )

    # Core behavior rules
    parts.append(CORE_BEHAVIOR_RULES)

    # Deep thinking
    if thinking:
        parts.append(
            "=== REASONING MODE ===\n"
            "Work through this step by step before answering:\n"
            "1. State the key assumptions you're making.\n"
            "2. Break the problem into the smallest useful sub-steps.\n"
            "3. Work through each sub-step, showing your reasoning.\n"
            "4. Sanity-check your result against the original question — does "
            "it actually answer what was asked?\n"
            "5. Then give the final answer clearly marked with 'ANSWER:'.\n"
            "=== END REASONING MODE ==="
        )

    # Tool policy
    if tools:
        parts.append(
            "=== TOOL POLICY ===\n"
            "When tool results are provided, use them precisely and cite what "
            "they actually returned. Do not paraphrase tool output into "
            "something it didn't say. If a tool call failed or returned "
            "nothing useful, say so explicitly instead of guessing.\n"
            "=== END TOOL POLICY ==="
        )

    # Deep research note
    if getattr(state, "deep_research_enabled", False):
        parts.append(
            "=== DEEP RESEARCH MODE ===\n"
            "You are in deep research mode. You have been provided with a research brief "
            "from multiple sources. Use it as the basis for your answer, but you may "
            "supplement with your own knowledge. Cite the key sources.\n"
            "=== END DEEP RESEARCH MODE ==="
        )

    return "\n\n".join(parts)

# ── STATE PERSISTENCE ─────────────────────────────────────────
def load_proj():
    if PROJ_FILE.exists():
        try:
            return json.loads(PROJ_FILE.read_text())
        except Exception:
            pass
    return {"Default":{"messages":[]}}
def save_proj(p):
    PROJ_FILE.write_text(json.dumps(p, indent=2))
def load_notes():
    if NOTES_FILE.exists():
        try:
            return json.loads(NOTES_FILE.read_text())
        except Exception:
            pass
    return {"Default":""}
def save_notes(n):
    NOTES_FILE.write_text(json.dumps(n, indent=2))

def init_state():
    check_dependencies(verbose=True)
    if not PROJ_FILE.exists():
        save_proj({"Default":{"messages":[]}})
    if not NOTES_FILE.exists():
        save_notes({"Default":""})
    if not KB_FILE.exists():
        save_kb([])
    logger.info("✅ init_state complete — local data files ready.")

def api_ctx(state, n=SHORT_CTX):
    return [{"role":m["role"],"content":m["content"]} for m in state.messages[-n:]]

def extract_steps(text):
    return [s.strip() for s in re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\n\n|\*\*ANSWER|\Z)", text, re.S) if len(s.strip())>10]

logger.info("✅ OmniX Core v9.1 loaded.")
print("✅ OmniX Core v9.1 loaded.")
