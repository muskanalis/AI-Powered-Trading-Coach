import os
import re
import time
import math
import datetime
import json
import requests
import shutil
import subprocess
import logging
from typing import Optional, Dict, Any
import threading, pathlib
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import yfinance as yf

# --- App & Database Configuration ---
app = Flask(__name__)

# Secret key for session signing (use env var in production)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# CORS: allow your frontend origin(s) and support credentials (cookies)
CORS(app, origins=["http://127.0.0.1:5173", "http://localhost:5173"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# MySQL configuration
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'Abhishek'
app.config['MYSQL_PASSWORD'] = 'Avengers/2005'
app.config['MYSQL_DB'] = 'smartcoach_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Optional session cookie settings
app.config.update(
    SESSION_COOKIE_SAMESITE=None,
    SESSION_COOKIE_SECURE=False
)
CORS(app, origins=["http://localhost:5173","http://127.0.0.1:5173"], supports_credentials=True)

bcrypt = Bcrypt(app)
mysql = MySQL(app)

# --- Caching System ---
api_cache = {}
CACHE_DURATION_SECONDS = 10


CHAT_LOG_DIR = os.path.join(os.path.dirname(__file__), "chat_logs")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)
CHAT_LOG_LOCK = threading.Lock()
MAX_LOG_MESSAGES = 2000
# --- Load Instrument List for Search ---
instrument_list = []
try:
    with open('instrument_list.json', 'r', encoding='utf-8') as f:
        instrument_list = json.load(f)
    print("Successfully loaded instrument list.")
except Exception as e:
    print(f"ERROR: Could not load instrument_list.json. Search will not work. {e}")

def _chat_log_path_for(user_id: Optional[int] = None, portfolio_id: Optional[int] = None) -> str:
    if portfolio_id:
        name = f"portfolio_{portfolio_id}.json"
    elif user_id:
        name = f"user_{user_id}.json"
    else:
        name = "anon.json"
    return os.path.join(CHAT_LOG_DIR, name)

def load_chat_logs(user_id: Optional[int] = None, portfolio_id: Optional[int] = None):
    path = _chat_log_path_for(user_id, portfolio_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load chat logs %s: %s", path, e)
            return []
    return []

def save_chat_logs(logs, user_id: Optional[int] = None, portfolio_id: Optional[int] = None):
    path = _chat_log_path_for(user_id, portfolio_id)
    try:
        # limit size
        if isinstance(logs, list) and len(logs) > MAX_LOG_MESSAGES:
            logs = logs[-MAX_LOG_MESSAGES:]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error("Failed to save chat logs %s: %s", path, e)
        return False

def append_chat_log(entry: dict, user_id: Optional[int] = None, portfolio_id: Optional[int] = None):
    path = _chat_log_path_for(user_id, portfolio_id)
    with CHAT_LOG_LOCK:
        logs = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(entry)
        # trim
        if len(logs) > MAX_LOG_MESSAGES:
            logs = logs[-MAX_LOG_MESSAGES:]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to append to chat log %s: %s", path, e)

# --- Helper: safe JSON input ---
def get_json_data():
    try:
        return request.get_json() or {}
    except Exception:
        return {}


def compute_coach_class_for(cur, portfolio_id, ticker, trade_type, price,
                            recent_n=10, pct_threshold=0.03):
    """
    Classify trade as NORMAL, FOMO, or PANIC based on recent prices.
    """
    cur.execute("""
        SELECT price
        FROM transactions
        WHERE portfolio_id=%s AND ticker=%s
        ORDER BY id DESC
        LIMIT %s
    """, (portfolio_id, ticker, recent_n))
    rows = cur.fetchall() or []

    if rows:
        total = sum(float(r.get('price')) if isinstance(r, dict) else float(r[0]) for r in rows)
        count = len(rows)
        recent_avg = (total / count) if count else price
    else:
        recent_avg = price

    delta_pct = (price - recent_avg) / recent_avg if recent_avg else 0.0

    t = (trade_type or "").upper()
    if t == "BUY" and delta_pct > pct_threshold:
        return "FOMO"
    if t == "SELL" and delta_pct < -pct_threshold:
        return "PANIC"
    return "NORMAL"

def require_auth():
    if "user" not in session:
        return False
    return True

def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Calls local Ollama model (must be running in background).
    Example: ollama pull llama3
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        output = result.stdout.decode("utf-8").strip()
        if not output:
            output = result.stderr.decode("utf-8").strip()
        return output
    except Exception as e:
        return f"Ollama error: {e}"

# --- User & Auth Endpoints ---
@app.route("/api/me", methods=["GET"])
def get_current_user():
    user = session.get("user")
    if user:
        return jsonify({"id": user.get("id"), "username": user.get("username")}), 200
    return jsonify({"error": "Not logged in"}), 401

@app.route("/api/register", methods=['POST'])
def register_user():
    data = get_json_data()
    username = (data.get('username') or "").strip()
    password = (data.get('password') or "")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return jsonify({"error": "Username already exists"}), 409
        cur.execute("INSERT INTO users(username, password) VALUES (%s, %s)", (username, hashed_password))
        mysql.connection.commit()
        return jsonify({"message": "User registered successfully!"}), 201
    finally:
        cur.close()

@app.route("/api/login", methods=['POST'])
def login_user():
    data = get_json_data()
    username = (data.get('username') or "").strip()
    password = (data.get('password') or "")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        db_hash = user.get('password') if isinstance(user, dict) else user[2]
        if not db_hash or not bcrypt.check_password_hash(db_hash, password):
            return jsonify({"error": "Invalid username or password"}), 401

        session.clear()
        session['user'] = {
            "id": user.get('id') if isinstance(user, dict) else user[0],
            "username": user.get('username') if isinstance(user, dict) else username
        }
        return jsonify({"message": "Login successful!"}), 200
    finally:
        cur.close()


# --- Stock Data API Endpoints ---
@app.route("/api/stock/<ticker>")
def get_stock_detail(ticker):
    full_ticker = f"{ticker}.NS"
    cache_key = full_ticker
    current_time = time.time()

    if cache_key in api_cache and current_time - api_cache[cache_key]['timestamp'] < CACHE_DURATION_SECONDS:
        return jsonify(api_cache[cache_key]['data'])

    try:
        stock = yf.Ticker(full_ticker)
        hist_data = stock.history(period="1d", interval="1m")
        if hist_data.empty:
            raise ValueError("No intraday data found")
        history = [{"time": index.strftime('%H:%M'), "price": round(row['Close'], 2)}
                   for index, row in hist_data.iterrows()]

        info = stock.info
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        previous_close = info.get('previousClose', price)
        change = price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0

        full_details = {
            "ticker": ticker,
            "name": info.get('longName', ticker),
            "price": round(price, 2),
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "history": history
        }
        api_cache[cache_key] = {'timestamp': current_time, 'data': full_details}
        return jsonify(full_details)
    except Exception as e:
        print(f"ERROR fetching data for {full_ticker}: {e}")
        return jsonify({"error": f"Failed to fetch live API data for {ticker}"}), 500


@app.route("/api/stocks")
def get_stocks():
    # Combined list (your original + extra popular stocks)
    default_stocks = [
        {"ticker": "RELIANCE", "name": "Reliance Industries"},
        {"ticker": "TCS", "name": "Tata Consultancy"},
        {"ticker": "HDFCBANK", "name": "HDFC Bank"},
        {"ticker": "INFY", "name": "Infosys"},
        {"ticker": "ICICIBANK", "name": "ICICI Bank"},
        {"ticker": "SBIN", "name": "State Bank of India"},
        {"ticker": "ITC", "name": "ITC Limited"},
        {"ticker": "TATAMOTORS", "name": "Tata Motors"},
        {"ticker": "WIPRO", "name": "Wipro"},
        {"ticker": "SUNPHARMA", "name": "Sun Pharma"},
        {"ticker": "ASIANPAINT", "name": "Asian Paints"},
        {"ticker": "BAJFINANCE", "name": "Bajaj Finance"},
        {"ticker": "LT", "name": "Larsen & Toubro"}
    ]
    return jsonify(default_stocks)


@app.route("/api/search/<query>")
def search_stocks(query):
    search_term = query.lower()
    results = []
    if not instrument_list:
        return jsonify([])
    for instrument in instrument_list:
        try:
            if not isinstance(instrument, dict):
                continue
            name = instrument.get('name')
            symbol = instrument.get('symbol')
            if not isinstance(name, str) or not isinstance(symbol, str):
                continue
            if search_term in name.lower() and symbol.upper().endswith('-EQ'):
                results.append({"ticker": symbol.replace('-EQ', ''), "name": name.title()})
            if len(results) >= 20:
                break
        except Exception:
            continue
    return jsonify(results)


# --- TODO: Portfolio & Transactions API (already in your full code) ---
# Keep the rest of your existing routes for portfolios, holdings, trade, logout etc.
# I left them out here only to shorten this reply — DO NOT delete them in your file.
# Just keep your original portfolio/transactions section under this point.

# --- Portfolio & Transactions API Endpoints ---

@app.route("/api/portfolios", methods=["GET"])
def get_portfolios():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT * FROM portfolios WHERE user_id = %s", (session["user"]["id"],))
        portfolios = cur.fetchall()
        return jsonify(portfolios)
    finally:
        cur.close()

@app.route("/api/portfolios", methods=["POST"])
def create_portfolio():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = get_json_data()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Portfolio name required"}), 400
    cur = mysql.connection.cursor()
    try:
        cur.execute("INSERT INTO portfolios (user_id, name) VALUES (%s, %s)", (session["user"]["id"], name))
        mysql.connection.commit()
        return jsonify({"message": "Portfolio created successfully!"}), 201
    finally:
        cur.close()

@app.route("/api/admin/reclass-transactions", methods=["POST"])
def reclass_transactions():
    # protect this endpoint in production (admin-only)
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT id, portfolio_id, ticker, type, quantity, price FROM transactions")
        rows = cur.fetchall() or []
        count = 0
        for r in rows:
            # r is dict (DictCursor)
            tid = r.get('id')
            pid = r.get('portfolio_id')
            tkr = r.get('ticker')
            ttype = r.get('type')
            price = float(r.get('price') or 0)
            # compute new coach_class with the same heuristic used in trade_stock
            # e.g. compute recent average etc (or call a shared helper function)
            new_label = compute_coach_class_for(cur, pid, tkr, ttype, price)
            cur.execute("UPDATE transactions SET coach_class=%s WHERE id=%s", (new_label, tid))
            count += 1
        mysql.connection.commit()
        return jsonify({"updated": count}), 200
    finally:
        cur.close()

@app.route("/api/portfolio/<int:portfolio_id>", methods=["GET"])
def get_portfolio(portfolio_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT * FROM portfolios WHERE id=%s AND user_id=%s", (portfolio_id, session["user"]["id"]))
        portfolio = cur.fetchone()
        if not portfolio:
            return jsonify({"error": "Portfolio not found"}), 404
        cur.execute("SELECT * FROM transactions WHERE portfolio_id=%s ORDER BY timestamp DESC", (portfolio_id,))
        transactions = cur.fetchall()
        return jsonify({"portfolio": portfolio, "transactions": transactions})
    finally:
        cur.close()

@app.route("/api/portfolio/<int:portfolio_id>/trade", methods=["POST"])
def trade_stock(portfolio_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = get_json_data()
    ticker = (data.get("ticker") or "").upper()
    trade_type = (data.get("type") or "").upper()

    try:
        quantity = int(data.get("quantity") or 0)
        price = float(data.get("price") or 0)
    except Exception:
        return jsonify({"error": "Invalid quantity/price"}), 400
     
    if not ticker or trade_type not in ("BUY", "SELL") or quantity <= 0 or price <= 0:
        return jsonify({"error": "Invalid trade data"}), 400

    cur = mysql.connection.cursor()
    try:
        # ownership check
        cur.execute("SELECT id FROM portfolios WHERE id=%s AND user_id=%s", (portfolio_id, session["user"]["id"]))
        if not cur.fetchone():
            return jsonify({"error": "Portfolio not found"}), 404

        # --- Heuristic classification ---
        coach_class = compute_coach_class_for(cur, portfolio_id, ticker, trade_type, price)

        # 1) compute recent average price for the ticker from last few transactions in this portfolio
        cur.execute("""
            SELECT price, quantity, type
            FROM transactions
            WHERE portfolio_id=%s AND ticker=%s
            ORDER BY id DESC
            LIMIT 10
        """, (portfolio_id, ticker))
        recent = cur.fetchall() or []

        # compute simple average of last N trade prices (weighted by qty might be better)
        if recent:
            total = 0.0
            count = 0
            for r in recent:
                p = float(r.get("price", 0) if isinstance(r, dict) else r[0])
                total += p
                count += 1
            recent_avg = (total / count) if count else price
        else:
            recent_avg = price

        # set thresholds
        PERCENT_THRESHOLD = 0.03  # 3%

        # if BUY and price is significantly above recent average => FOMO
        if trade_type == "BUY" and (price - recent_avg) / recent_avg > PERCENT_THRESHOLD:
            coach_class = "FOMO"

        # if SELL and price is significantly below recent average => PANIC
        if trade_type == "SELL" and (price - recent_avg) / recent_avg < -PERCENT_THRESHOLD:
            coach_class = "PANIC"

        # (OPTIONAL) Use user emotional_score to bias classification:
        # cur.execute("SELECT emotional_score FROM users WHERE id = %s", (session['user']['id'],))
        # u = cur.fetchone()
        # if u and u.get('emotional_score', 50) > 70 and coach_class == 'FOMO':
        #     coach_class = 'FOMO'  # emphasize label for anxious users, etc.

        # --- Insert transaction with coach_class and return the inserted row ---
        cur.execute(
            "INSERT INTO transactions (portfolio_id, ticker, type, quantity, price, coach_class) VALUES (%s,%s,%s,%s,%s,%s)",
            (portfolio_id, ticker, trade_type, quantity, price, coach_class)
        )
        mysql.connection.commit()

        cur.execute("SELECT * FROM transactions WHERE id = LAST_INSERT_ID()")
        tx = cur.fetchone()
        return jsonify({"transaction": tx}), 201

    except Exception as e:
        print("Trade DB error:", e)
        return jsonify({"error": "An error occurred processing the trade."}), 500
    finally:
        cur.close()

@app.route("/debug/session", methods=["GET", "POST"])
def debug_session():
    # show what the server sees (headers + session contents)
    from flask import make_response
    info = {
        "headers": dict(request.headers),
        "cookies_sent_by_browser": request.cookies,   # cookies Flask received in request
        "session_on_server": session.get("user")
    }
    resp = make_response(jsonify(info), 200)
    # include the same CORS headers – Flask-CORS should set these already
    return resp

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"}), 200

@app.route("/api/portfolio/<int:portfolio_id>/holdings", methods=["GET"])
def get_holdings(portfolio_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    cur = mysql.connection.cursor()
    try:
        # verify ownership
        cur.execute("SELECT id FROM portfolios WHERE id=%s AND user_id=%s", (portfolio_id, session["user"]["id"]))
        if not cur.fetchone():
            return jsonify({"error": "Portfolio not found"}), 404

        # Compute average cost safely: only BUY transactions count for avg_cost
        cur.execute("""
            SELECT 
                ticker,
                SUM(CASE WHEN type='BUY' THEN quantity ELSE 0 END) AS total_bought,
                SUM(CASE WHEN type='SELL' THEN quantity ELSE 0 END) AS total_sold,
                SUM(CASE WHEN type='BUY' THEN quantity*price ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN type='BUY' THEN quantity ELSE 0 END), 0) AS avg_buy_price
            FROM transactions
            WHERE portfolio_id=%s
            GROUP BY ticker
        """, (portfolio_id,))

        rows = cur.fetchall()
        holdings = []
        for r in rows:
            total_bought = r.get("total_bought") or 0
            total_sold = r.get("total_sold") or 0
            qty = total_bought - total_sold
            if qty <= 0:
                continue  # skip fully sold positions
            avg_cost = r.get("avg_buy_price") or 0.0
            holdings.append({
                "ticker": r["ticker"],
                "quantity": qty,
                "avg_cost": round(abs(avg_cost), 2)  # ensure positive
            })

        return jsonify(holdings)
    finally:
        cur.close()

@app.route("/api/portfolio/<int:portfolio_id>", methods=["DELETE"])
def delete_portfolio(portfolio_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    cur = mysql.connection.cursor()
    try:
        cur.execute("DELETE FROM portfolios WHERE id=%s AND user_id=%s", (portfolio_id, session["user"]["id"]))
        mysql.connection.commit()
        return jsonify({"message": "Portfolio deleted"}), 200
    finally:
        cur.close()

# --- Hugging Face / OpenAI AI Coach (robust implementation) ---
# Uses requests to call Hugging Face Inference API and OpenAI REST API.
# Make sure to set HF_API_TOKEN and OPENAI_API_KEY environment variables in the same shell that runs Flask.

# --- Simple, robust AI Coach (tries external LLMs, otherwise local fallback) ---

HF_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Try small external providers (we will attempt and catch errors)
def try_hf(model: str, prompt: str):
    if not HF_TOKEN:
        raise RuntimeError("HF token not set")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.2}}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"HuggingFace error {r.status_code}: {r.text[:800]}")
    j = r.json()
    # Prefer common 'generated_text' responses
    if isinstance(j, list) and len(j) and isinstance(j[0], dict):
        return j[0].get("generated_text", "") or str(j)
    if isinstance(j, dict) and "generated_text" in j:
        return j["generated_text"]
    return json.dumps(j)

def try_openai(prompt: str):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise AI trading coach. No legal/regulatory advice."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:800]}")
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(j)

# A tiny deterministic fallback "coach" so the endpoint ALWAYS replies
def fallback_coach(user_msg: str, extra: dict | None = None) -> str:
    """
    Simple deterministic responses:
     - If user asks about a ticker or includes 'TCS' or 'INFY' -> give generic checklists
     - If user asks 'how am I doing' -> provide behavioural checklist
     - Otherwise, give concise practical tips
    """
    msg = user_msg.lower()
    # detect tickers (approx): TCS, INFY, RELIANCE, HDFCBANK, SBIN, TATAMOTORS etc.
    tickers = re.findall(r'\b([A-Z]{2,6})\b', user_msg)
    # fallback heuristics
    if any(w in msg for w in ["how am i doing", "how am i trading", "am i doing well"]):
        return (
            "Quick assessment checklist:\n"
            "1) Position sizing: ensure no single stock > 3-5% of portfolio unless deliberate.\n"
            "2) Stop-loss: set stop levels before entering and keep them discipline-driven.\n"
            "3) Journal: note entry thesis + target + stop; review trades weekly.\n"
            "4) Diversification: check sector concentration (>30% in one sector -> rebalance).\n"
            "5) Emotions: if frequent intraday flipping, consider longer timeframe or rules.\n"
            "If you share a ticker, buy price and current price I can give concrete next steps."
        )

    if tickers:
        t = tickers[0].upper()
        return (
            f"Checklist for {t}:\n"
            "• Re-evaluate thesis: why you bought (growth, margin, news?).\n"
            "• Check volume: rising volume on price up is supportive; falling volume on rally is weak.\n"
            "• Risk control: reduce size or set stop if position >3-5% of capital.\n"
            "• Plan exit: set target and stop; avoid emotional exits.\n"
        )

    # ask clarifying question if user is vague
    if len(msg.split()) < 6:
        return "Can you give one specific example (ticker + buy price + current price) or say 'how am I doing'?"

    # default practical tips
    return (
        "Practical tips:\n"
        "1) Use position sizing and stop-loss rules before entering.\n"
        "2) Keep a simple watchlist with reasons to buy/sell.\n"
        "3) If you see FOMO: reduce size, revisit thesis.\n"
        "4) If panic-selling: check if your stop was hit or if fundamentals changed.\n"
    )
# --- Local small-model fallback using transformers ---
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "distilgpt2")  # small local model

_local_model = None
_local_tokenizer = None

def ensure_local_model_loaded():
    global _local_model, _local_tokenizer
    if _local_model is not None and _local_tokenizer is not None:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        # Load tokenizer + model into memory (CPU)
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        _local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
        # keep a simple generator pipeline
        # set device to CPU (default) — if you have a GPU and torch detects it, it will be used
        # NOTE: this step will download model files the first time (needs internet)
    except Exception as e:
        print("Local model load failed:", e)
        _local_model = None
        _local_tokenizer = None

def generate_local(prompt: str, max_new_tokens: int = 150):
    """
    Generate via local HuggingFace model. Uses transformers.pipeline to
    be robust across versions and handle device placement.
    """
    ensure_local_model_loaded()
    if not _local_model or not _local_tokenizer:
        raise RuntimeError("Local model not available")

    try:
        # Use the pipeline helper which manages device placement and tokenization.
        from transformers import pipeline
        # device = 0 for first GPU, -1 for CPU. We'll default to CPU to be safe.
        pipe = pipeline("text-generation", model=_local_model, tokenizer=_local_tokenizer, device=-1)
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, temperature=0.6)
        if isinstance(out, list) and len(out) > 0:
            # many generation outputs contain 'generated_text'
            text = out[0].get("generated_text") if isinstance(out[0], dict) else str(out[0])
            return text or str(out)
        return str(out)
    except Exception as e:
        # bubble up so caller can fall back
        raise RuntimeError(f"Local generation failed: {e}")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def find_ollama_executable() -> str | None:
    """
    Find ollama executable path. Priority:
     1) explicit env var OLLAMA_BIN
     2) shutil.which('ollama')
     3) some common Windows install locations
    """
    env_path = os.getenv("OLLAMA_BIN")
    if env_path and os.path.exists(env_path):
        return env_path

    exe = shutil.which("ollama")
    if exe:
        return exe

    candidates = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
        os.path.expanduser(r"~\AppData\Local\Ollama\ollama.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def call_ollama(prompt: str, model: str = "llama3", timeout: int = 30) -> str:
    """
    Run `ollama run <model>` and send prompt to stdin.
    Returns stdout if available, otherwise raises a RuntimeError with helpful message.
    """
    exe = find_ollama_executable()
    if not exe:
        raise RuntimeError(
            "Ollama executable not found. Solutions:\n"
            " - Start Flask from the same shell where `ollama` works,\n"
            " - Or set OLLAMA_BIN to the full path: setx OLLAMA_BIN \"C:\\full\\path\\to\\ollama.exe\"\n"
            f"Current PATH head: {os.environ.get('PATH','')[:800]}"
        )

    # run the process: pass prompt to stdin
    try:
        proc = subprocess.run(
            [exe, "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama call timed out.")
    except FileNotFoundError:
        raise RuntimeError(f"Ollama not found at configured path: {exe}")

    stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
    stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
    logger.info("call_ollama rc=%s stdout_len=%d stderr_len=%d", proc.returncode, len(stdout), len(stderr))

    if stdout:
        return stdout
    if stderr:
        raise RuntimeError(f"Ollama ran but produced stderr:\n{stderr}")
    raise RuntimeError("Ollama returned empty output.")

def fetch_portfolio_summary_for_user(user_id):
    """
    Returns simple portfolio summary { cash: float, holdings: [ {ticker, quantity, avg_cost, current_price} ] }
    """
    cur = mysql.connection.cursor()
    try:
        # 1) Find user's default portfolio (you might want to allow selection instead)
        cur.execute("SELECT id FROM portfolios WHERE user_id=%s ORDER BY id LIMIT 1", (user_id,))
        p = cur.fetchone()
        if not p:
            return {"cash": 0.0, "holdings": []}
        portfolio_id = p.get("id")

        # 2) Compute cash (if you track cash) — fallback 0
        # If you have a 'cash' column in portfolios change this query accordingly.
        cur.execute("SELECT COALESCE(SUM(CASE WHEN type='SELL' THEN quantity*price WHEN type='BUY' THEN -quantity*price ELSE 0 END),0) as cash_flow FROM transactions WHERE portfolio_id=%s", (portfolio_id,))
        cash_row = cur.fetchone() or {}
        cash_flow = float(cash_row.get("cash_flow") or 0.0)

        # NOTE: The above cash_flow is a simple heuristic. If you track deposits/withdraws, use that table.

        # 3) Build holdings: sum quantities and compute avg_cost
        cur.execute("""
            SELECT ticker,
                   SUM(CASE WHEN type='BUY' THEN quantity WHEN type='SELL' THEN -quantity ELSE 0 END) AS qty,
                   SUM(CASE WHEN type='BUY' THEN quantity*price WHEN type='SELL' THEN -quantity*price ELSE 0 END) AS cost_basis
            FROM transactions
            WHERE portfolio_id=%s
            GROUP BY ticker
            HAVING qty <> 0
        """, (portfolio_id,))
        rows = cur.fetchall() or []

        holdings = []
        for r in rows:
            qty = float(r.get("qty") or 0)
            cost_basis = float(r.get("cost_basis") or 0.0)
            avg_cost = (cost_basis / qty) if qty else 0.0
            # Optional: look up latest price using your caching yfinance endpoint:
            # call get_stock_detail or use yfinance directly (we'll try yfinance quickly)
            latest_price = None
            try:
                full_ticker = f"{r['ticker']}.NS"
                t = yf.Ticker(full_ticker)
                info = t.info
                latest_price = info.get('currentPrice', info.get('regularMarketPrice', None))
                # fallback use avg_cost
                if latest_price is None:
                    latest_price = round(avg_cost, 2)
            except Exception:
                latest_price = round(avg_cost, 2)

            holdings.append({
                "ticker": r["ticker"],
                "quantity": int(qty),
                "avg_cost": round(avg_cost, 2),
                "current_price": round(float(latest_price or avg_cost), 2)
            })

        return {"cash": round(float(cash_flow or 0.0), 2), "holdings": holdings}
    finally:
        cur.close()

@app.route("/api/coach/chat", methods=["POST"])
def ai_coach_chat():
     """
    Accepts: { message: str, optional portfolio_summary: { cash, holdings: [...] }, optional portfolio_id }
    Behavior:
      - Prefer portfolio_summary from request body (frontend).
      - If not provided and user logged in, fetch DB summary.
      - Compute numeric summary (totals) and append short summary to LLM prompt.
      - Try Ollama (if available) else fallback to simple_coach_reply.
      - Append both user and assistant messages to server-side chat logs.
      - Return reply + portfolio_summary + computed_summary for front-end debugging.
    """
     data = request.get_json() or {}
     user_msg = (data.get("message") or "").strip()
     if not user_msg:
        return jsonify({"error": "message required"}), 400

     # --- identify user and optional portfolio ---
     user_id = session.get("user", {}).get("id")
     portfolio_id = data.get("portfolio_id") or data.get("portfolioId")
     if isinstance(portfolio_id, str) and portfolio_id.isdigit():
        portfolio_id = int(portfolio_id)
     elif isinstance(portfolio_id, (int, float)):
        portfolio_id = int(portfolio_id)
     else:
        portfolio_id = None

     # 1) Prefer portfolio_summary sent in request body
     portfolio_summary = data.get("portfolio_summary")

     # 2) If not provided by client, try to fetch server-side summary for logged-in user
     if not portfolio_summary and "user" in session:
        try:
            portfolio_summary = fetch_portfolio_summary_for_user(session["user"]["id"])
        except Exception:
            portfolio_summary = None

     # 3) Always compute a normalized numeric summary (cash, holdings value, total)
     computed_summary = None
     try:
        computed_summary = compute_portfolio_summary(portfolio_summary) if portfolio_summary else None
     except Exception:
        computed_summary = None

     # Build a clear prompt for the LLM; include computed_summary if available
     prompt = "You are an AI trading coach. Be concise and practical (no legal/financial advice).\n\n"
     if computed_summary:
        holdings_lines = []
        for h in computed_summary.get("holdings_summary", []):
            holdings_lines.append(
                f"{h['ticker']}: qty={h['quantity']}, avg_cost={h['avg_cost']}, current_price={h['current_price']}, value={h['value']:.2f}"
            )
        prompt += "Portfolio summary (computed):\n"
        prompt += f"Cash: {_format_inr(computed_summary.get('cash', 0))}\n"
        prompt += "Holdings:\n" + ("\n".join(holdings_lines) if holdings_lines else "none") + "\n\n"
     prompt += "User: " + user_msg + "\nAssistant:"

     # Prepare a user log entry (we'll append to server logs later)
     user_entry = {"role": "user", "text": user_msg, "ts": datetime.datetime.utcnow().isoformat() + "Z"}

     # Try Ollama (local) first if configured
     reply_text = None
     source = None
     model = None
     try:
        model = os.getenv("OLLAMA_MODEL", "llama3")
        reply = call_ollama(prompt, model=model, timeout=40)
        reply_text = reply
        source = "ollama"
     except Exception as e:
        logger.warning("Ollama attempt failed: %s", str(e))
        # fall through to fallback

     # If we don't have a reply from Ollama, use the simple / fallback coach
     if reply_text is None:
        try:
            result = simple_coach_reply(user_msg, portfolio_summary=portfolio_summary)
            reply_text = result.get("reply")
            # prefer returned summary if present
            reply_summary = result.get("summary", computed_summary)
            source = "simple"
        except Exception:
            reply_text = "Sorry — couldn't generate coach reply."
            reply_summary = computed_summary
            source = "simple"
     else:
        reply_summary = computed_summary

     # Build assistant log entry
     assistant_entry = {
        "role": "assistant",
        "text": reply_text,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "source": source
     }
     if model:
        assistant_entry["model"] = model

     # Append both user + assistant entries to server logs (safe: don't fail the request on logging errors)
     try:
        append_chat_log(user_entry, user_id=user_id, portfolio_id=portfolio_id)
        append_chat_log(assistant_entry, user_id=user_id, portfolio_id=portfolio_id)
     except Exception as e:
        logger.warning("Failed to append chat logs: %s", e)

     # Return the response payload (keep compatibility)
     resp = {
        "source": source,
        "reply": reply_text,
        "portfolio_summary": portfolio_summary,
        "computed_summary": reply_summary
     }
     if model:
        resp["model"] = model

     return jsonify(resp), 200

@app.route("/api/coach/logs", methods=["GET", "POST"])
def coach_logs_endpoint():
    """
    GET -> returns logs for this session user or anon (optionally ?portfolio_id=)
    POST -> body { logs: [...], portfolio_id?: int, clear?: bool }
            saves logs to server (overwrite) or clear if clear:true.
    """
    user = session.get("user")
    user_id = user.get("id") if user else None

    if request.method == "GET":
        portfolio_id = request.args.get("portfolio_id") or request.args.get("portfolioId")
        if portfolio_id and str(portfolio_id).isdigit():
            portfolio_id = int(portfolio_id)
        else:
            portfolio_id = None
        logs = load_chat_logs(user_id=user_id, portfolio_id=portfolio_id)
        return jsonify({"logs": logs}), 200

    # POST
    data = get_json_data()
    logs = data.get("logs", None)
    portfolio_id = data.get("portfolio_id") or data.get("portfolioId")
    clear_flag = bool(data.get("clear", False))
    if portfolio_id and str(portfolio_id).isdigit():
        portfolio_id = int(portfolio_id)
    else:
        portfolio_id = None

    if clear_flag:
        path = _chat_log_path_for(user_id, portfolio_id)
        try:
            if os.path.exists(path):
                os.remove(path)
            return jsonify({"saved": True, "cleared": True}), 200
        except Exception as e:
            return jsonify({"error": "failed clearing logs", "detail": str(e)}), 500

    if logs is None:
        return jsonify({"error": "logs required"}), 400
    if not isinstance(logs, list):
        return jsonify({"error": "logs must be an array"}), 400

    # limit
    if len(logs) > MAX_LOG_MESSAGES:
        logs = logs[-MAX_LOG_MESSAGES:]

    ok = save_chat_logs(logs, user_id=user_id, portfolio_id=portfolio_id)
    if not ok:
        return jsonify({"error": "failed saving logs"}), 500
    return jsonify({"saved": True, "count": len(logs)}), 200

@app.route("/debug/ollama", methods=["GET"])
def debug_ollama():
    return jsonify({
        "shutil_which": shutil.which("ollama"),
        "OLLAMA_BIN_env": os.getenv("OLLAMA_BIN"),
        "PATH_head": os.environ.get("PATH", "")[:1200]
    })
  
def _normalize_ticker_list(text: str):
    return list({m.group(1).upper() for m in re.finditer(r'\b([A-Z]{2,6})\b', text)})

def _pct(x, y):
    try:
        return (x - y) / y if y else 0.0
    except Exception:
        return 0.0

def _format_inr(x):
    try:
        x = float(x)
    except Exception:
        return "₹0"
    # simple formatting with thousands separators and 2 decimals
    return "₹{:,.2f}".format(x)

def _parse_number(x):
    """
    Robust parse: accepts numbers, strings with commas, currency symbols,
    parentheses for negatives, and returns float.
    Examples: "1,234.56" -> 1234.56, "(953.82)" -> -953.82, "95382" -> 95382.0
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    # remove currency symbols and spaces
    s = re.sub(r'[₹$,]', '', s)
    # parentheses => negative
    if re.match(r'^\(.*\)$', s):
        s = '-' + s.strip('()')
    # handle common thousands separators and stray chars
    s = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(s)
    except Exception:
        # last resort: try integer conversion
        try:
            return float(int(re.sub(r'\D', '', s) or 0))
        except Exception:
            return 0.0

def compute_portfolio_summary(portfolio_summary: dict | None):
    """
    Input expected:
      { cash: number|string, holdings: [ { ticker, quantity, avg_cost, current_price } ] }
    Returns numeric totals and per-holding breakdown with pct_of_portfolio.
    """
    if not portfolio_summary or not isinstance(portfolio_summary, dict):
        return {"total_value": 0.0, "cash": 0.0, "holdings_summary": []}

    cash = _parse_number(portfolio_summary.get("cash", 0))

    holdings_in = portfolio_summary.get("holdings", []) or []
    holdings_summary = []
    total_holdings_value = 0.0

    for h in holdings_in:
        # Support both dict and list-like representations defensively
        if isinstance(h, dict):
            ticker = (h.get("ticker") or "").upper()
            qty = _parse_number(h.get("quantity", 0))
            avg_cost = _parse_number(h.get("avg_cost", h.get("avgCost", 0)))
            current_price = _parse_number(h.get("current_price", h.get("currentPrice", h.get("current", avg_cost))))
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            ticker = str(h[0]).upper()
            qty = _parse_number(h[1])
            avg_cost = _parse_number(h[2] if len(h) > 2 else 0)
            current_price = _parse_number(h[3] if len(h) > 3 else avg_cost)
        else:
            continue

        # Defensive: if avg_cost or current_price look like they are in paise (too big),
        # try to detect and normalize
        # (heuristic) if avg_cost > 1e6 or current_price > 1e6, divide by 100
        if avg_cost and avg_cost > 1e6:
            avg_cost = avg_cost / 100.0
        if current_price and current_price > 1e6:
            current_price = current_price / 100.0

        # If avg_cost is negative (likely a bug), take absolute and log or mark
        if avg_cost < 0:
            avg_cost = abs(avg_cost)

        value = max(0.0, qty * current_price)
        total_holdings_value += value

        holdings_summary.append({
            "ticker": ticker or "UNKNOWN",
            "quantity": qty,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "value": value
        })

    total_value = cash + total_holdings_value
    # compute percentages safely
    for h in holdings_summary:
        h["pct_of_portfolio"] = (100.0 * h["value"] / total_value) if total_value else 0.0

    return {
        "total_value": total_value,
        "cash": cash,
        "holdings_summary": holdings_summary
    }
    
def simple_coach_reply(user_msg: str, portfolio_summary: dict | None = None):
    """
    Enhanced simple coach:
     - computes portfolio summary if provided
     - returns dict with 'reply' (text), 'tags' and the numeric summary under 'summary'
    """
    um = (user_msg or "").strip()
    if not um:
        return {"reply": "Please type a question or provide ticker + buy price + current price (e.g. 'TCS 3500 3620').", "tags": ["clarify"]}

    # compute portfolio summary (if any) so we can use numeric values in answers
    summary = compute_portfolio_summary(portfolio_summary)

    low = um.lower()
    # direct "how am I doing" assessment
    if any(phrase in low for phrase in ["how am i doing", "how am i trading", "am i doing well", "how is my performance"]):
        advice = [
            "Quick assessment checklist:",
            "1) Position sizing: avoid >5% per position unless it's core.",
            "2) Stop-loss: define entry, stop and target before trading.",
            "3) Diversify: check sector concentration (>30% in one sector -> rebalance).",
            "4) Journal: log entries and review weekly."
        ]

        # if we have holdings, add computed stats
        if summary["holdings_summary"]:
            # find top holding
            top = max(summary["holdings_summary"], key=lambda x: x["value"])
            top_pct = top.get("pct_of_portfolio", 0.0)
            advice.append(f"Top holding {top.get('ticker')} is ~{top_pct:.1f}% of portfolio value.")
            if top_pct > 25:
                advice.append("Consider trimming the top holding to reduce single-stock risk.")
        # friendly summary header
        total_str = _format_inr(summary["total_value"])
        cash_str = _format_inr(summary["cash"])
        reply_text = f"Portfolio summary: Total {total_str}, Cash {cash_str}.\n\n" + "\n".join(advice)
        return {"reply": reply_text, "tags": ["assessment"], "summary": summary}

    # ticker-specific checklist
    tickers = re.findall(r'\b([A-Z]{2,6})\b', user_msg)
    if tickers:
        t = tickers[0].upper()
        checklist = [
            f"Checklist for {t}:",
            "• Re-evaluate your original thesis (why you bought).",
            "• Check volume trend: rising volume on price increases is positive.",
            "• Size & risk: keep stop so you risk <1-2% of total capital per trade.",
            "• Set target & stop; avoid emotional exits."
        ]
        nums = [float(n) for n in re.findall(r'(?<![A-Z0-9])([0-9]+(?:\.[0-9]+)?)', user_msg)]
        if len(nums) >= 2:
            buy, curr = nums[0], nums[1]
            pct = (curr - buy) / buy if buy else 0.0
            if pct >= 0.05:
                checklist.append(f"Price is up {pct*100:.1f}% — consider booking some gains.")
            elif pct <= -0.05:
                checklist.append(f"Price is down {pct*100:.1f}% — check thesis & use rule-based stop-loss.")
            else:
                checklist.append("Price near entry — hold unless new negative info appears.")
        reply_text = "\n".join(checklist)
        return {"reply": reply_text, "tags": ["ticker"], "summary": summary}

    # behavioral/fomo/panic detection
    if any(w in low for w in ["fomo", "panic", "panic sell", "fear", "missed out"]):
        return {"reply": "If you feel FOMO: reduce size and re-check thesis. If panic-selling, verify if stop was hit or fundamentals changed.", "tags": ["behavior"], "summary": summary}

    # short how-to-trade guidance
    if any(w in low for w in ["how to trade", "strategy", "entry", "exit", "stop loss"]):
        tips = [
            "Short guide:",
            "1) Define entry hypothesis + stop + target.",
            "2) Risk per trade: keep to small % (1-2%).",
            "3) Use checklist: catalyst, trend, volume, valuation.",
            "4) Keep a trade journal and review weekly."
        ]
        return {"reply": "\n".join(tips), "tags": ["howto"], "summary": summary}

    if len(user_msg.split()) < 5:
        return {"reply": "Can you give a specific example (ticker + buy price + current price) or say 'How am I doing?'", "tags": ["clarify"], "summary": summary}

    default = [
        "Practical tips:",
        "• Use position sizing and stop-loss rules before entering.",
        "• If frequently switching positions, implement weekly review rules.",
        "• When uncertain, reduce size and diversify."
    ]
    return {"reply": "\n".join(default), "tags": ["general"], "summary": summary}


# Endpoint that uses the improved simple_coach_reply
@app.route("/api/coach/simple", methods=["POST"])
def ai_coach_simple():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    portfolio_summary = data.get("portfolio_summary")
    try:
        result = simple_coach_reply(msg, portfolio_summary=portfolio_summary)
    except Exception as e:
        result = {"reply": "Error generating advice", "tags": ["error"], "summary": {"total_value": 0, "cash": 0, "holdings_summary": []}}
    out = {
        "source": "simple",
        "reply": result.get("reply"),
        "tags": result.get("tags", []),
        "summary": result.get("summary", {}),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return jsonify(out), 200

@app.route("/debug/ollama-info", methods=["GET"])
def debug_ollama_info():
    found = shutil.which("ollama")
    return jsonify({
        "shutil_which": found,
        "env_path_head": os.environ.get("PATH", "")[:1000],
        "possible_locations": {
            "program_files": os.path.exists(r"C:\Program Files\Ollama\ollama.exe"),
            "user_appdata": os.path.exists(os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"))
        }
    })

# --- Learning progress endpoints (add to app.py) ---
LEARN_DIR = os.path.join(os.path.dirname(__file__), "learn_progress")
os.makedirs(LEARN_DIR, exist_ok=True)
LEARN_LOCK = threading.Lock()

def _learn_path_for(user_id: Optional[int] = None):
    if user_id:
        return os.path.join(LEARN_DIR, f"user_{user_id}_learn.json")
    return os.path.join(LEARN_DIR, "anon_learn.json")

def load_learn_progress(user_id: Optional[int] = None):
    path = _learn_path_for(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load learn progress %s: %s", path, e)
            return {}
    return {}

def save_learn_progress(progress: dict, user_id: Optional[int] = None):
    path = _learn_path_for(user_id)
    with LEARN_LOCK:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to save learn progress %s: %s", path, e)
            return False

@app.route("/api/learn/progress", methods=["GET", "POST"])
def learn_progress():
    """
    GET -> returns {"progress": {...}} for current user (or anon),
    POST -> body { progress: {...} } saves the progress for the current user.
    """
    user = session.get("user")
    user_id = user.get("id") if user else None

    if request.method == "GET":
        p = load_learn_progress(user_id)
        return jsonify({"progress": p}), 200

    # POST - save
    data = get_json_data()
    progress = data.get("progress")
    if progress is None:
        return jsonify({"error": "progress required"}), 400

    # normalize timestamp
    progress["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    ok = save_learn_progress(progress, user_id)
    if not ok:
        return jsonify({"error": "failed saving progress"}), 500
    return jsonify({"saved": True, "progress": progress}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
