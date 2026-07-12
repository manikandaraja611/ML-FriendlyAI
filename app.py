"""
AI Chatbot Application
======================
A smart Q&A chatbot that uses TF-IDF vectorization and cosine similarity
to find the best matching answer from a curated dataset of 1000+ Q&A pairs.

Supports both CLI and Flask web server modes.

Usage:
    CLI mode:   python app.py
    Web mode:   python app.py --web
"""

import os
import csv
import argparse
import re
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit import runtime


# ──────────────────────────────────────────────
#  Chatbot Engine
# ──────────────────────────────────────────────

class Chatbot:
    """A similarity-based chatbot using TF-IDF and cosine similarity."""

    CONFIDENCE_THRESHOLD = 0.25  # Minimum similarity score to return an answer

    def __init__(self, data_path: str = "data.csv"):
        """Load Q&A data and build the TF-IDF model."""
        self.data_path = data_path
        self.questions = []
        self.answers = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None
        self._load_data()
        self._build_model()

    def _load_data(self):
        """Load question-answer pairs from the CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip().str.lower()

        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")

        df.dropna(subset=["question", "answer"], inplace=True)
        self.questions = df["question"].astype(str).tolist()
        self.answers = df["answer"].astype(str).tolist()
        print(f"Loaded {len(self.questions)} Q&A pairs from '{self.data_path}'")

    def _build_model(self):
        """Fit the TF-IDF vectorizer on all questions."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        print("TF-IDF model built successfully")

    def get_response(self, user_input: str) -> dict:
        """
        Find the best matching answer for the user's input.

        Returns:
            dict with keys: 'answer', 'confidence', 'matched_question'
        """
        user_input = user_input.strip()
        if not user_input:
            return {
                "answer": "Please enter a valid question.",
                "confidence": 0.0,
                "matched_question": None,
            }

        # --- Check for simple rules (Math, Date, Time) ---
        input_lower = user_input.lower()
        
        # 1. Date check
        if re.search(r'\bdate\b', input_lower) and not re.search(r'\b(update|candidate)\b', input_lower):
            return {
                "answer": f"Today's date is {datetime.date.today().strftime('%B %d, %Y')}.",
                "confidence": 1.0,
                "matched_question": "Rule: Current Date",
            }
            
        # 2. Time check
        if re.search(r'\btime\b', input_lower) and not re.search(r'\b(space|complex)\b', input_lower):
            return {
                "answer": f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.",
                "confidence": 1.0,
                "matched_question": "Rule: Current Time",
            }

        # 3. Math check (e.g. 2+5=?)
        math_expr = user_input.replace("=", "").replace("?", "").strip()
        # Must contain at least one digit and only math-related characters
        if re.search(r'\d', math_expr) and re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', math_expr):
            try:
                # Safe eval since string only has digits and basic operators
                result = eval(math_expr, {"__builtins__": None}, {})
                # Format to remove .0 if it's an integer result
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                return {
                    "answer": f"The answer is {result}",
                    "confidence": 1.0,
                    "matched_question": "Rule: Math Calculation",
                }
            except Exception:
                pass
        # ------------------------------------------------

        # Vectorize user input and compute similarity
        user_vec = self.vectorizer.transform([user_input.lower()])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # Find the best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.CONFIDENCE_THRESHOLD:
            return {
                "answer": self.answers[best_idx],
                "confidence": round(best_score, 4),
                "matched_question": self.questions[best_idx],
            }
        else:
            return {
                "answer": "I'm sorry, I don't have an answer for that. Could you rephrase?",
                "confidence": round(best_score, 4),
                "matched_question": None,
            }


# ──────────────────────────────────────────────
#  CLI Mode
# ──────────────────────────────────────────────

def run_cli(bot: Chatbot):
    """Run the chatbot in interactive command-line mode."""
    print("\n" + "=" * 50)
    print("  🤖  Friendly AI  –  CLI Mode")
    print("=" * 50)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        result = bot.get_response(user_input)
        print(f"Bot: {result['answer']}")
        print(f"     (confidence: {result['confidence']:.2%})\n")


# ──────────────────────────────────────────────
#  Web (Flask) Mode
# ──────────────────────────────────────────────

def run_web(bot: Chatbot, host: str = "0.0.0.0", port: int = 5000):
    """Run the chatbot as a Flask web API."""
    from flask import Flask, request, jsonify, render_template_string

    app = Flask(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "flask_index.html"), "r", encoding="utf-8") as f:
        HTML_PAGE = f.read()

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        result = bot.get_response(user_message)
        return jsonify(result)

    print(f"\n🌐 Starting web server at http://localhost:{port}")
    app.run(host=host, port=port, debug=False)


# ──────────────────────────────────────────────
#  Streamlit Mode
# ──────────────────────────────────────────────

def run_streamlit():
    """Run the chatbot using Streamlit UI."""
    st.set_page_config(page_title="Friendly AI", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "streamlit_style.css"), "r", encoding="utf-8") as f:
        st_css = f.read()
    st.markdown(f"<style>{st_css}</style>", unsafe_allow_html=True)

    # Initialize bot
    if "bot" not in st.session_state:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data.csv")
        try:
            st.session_state.bot = Chatbot(data_path=data_path)
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            st.stop()
            
    bot = st.session_state.bot

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🤖 Friendly AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.info(f"**Total Questions:** {len(bot.questions)}")
        st.markdown("### ⚙️ System Status")
        st.success("🟢 Online & Ready")
        
        st.markdown("---")
        st.markdown("### 📁 Dataset")
        st.text("Offline CSV Loaded")
        
        st.markdown("---")
        # Clear Chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        # Export Chat
        if st.session_state.messages:
            chat_text = "\n".join([f"[{m['timestamp']}] {m['role'].upper()}: {m['raw_content']}" for m in st.session_state.messages])
            st.download_button(
                label="📥 Export Chat",
                data=chat_text,
                file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Main Header
    st.markdown('<div class="main-title">🤖 Friendly AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Offline Dataset Based AI</div>', unsafe_allow_html=True)

    # Display chat messages using custom HTML layout
    if st.session_state.messages:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.messages:
            role_class = "user" if msg["role"] == "user" else "ai"
            icon = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Friendly AI"
            
            conf_html = ""
            if msg["role"] == "assistant" and "confidence" in msg:
                conf_html = f"<div class='confidence-badge'>Confidence: {msg['confidence']*100:.1f}%</div>"
                
            chat_html += f'<div class="message-row {role_class}"><div class="bubble {role_class}"><div class="bubble-header">{icon}</div><div>{msg["content"]}</div>{conf_html}<div class="bubble-time">{msg["timestamp"]}</div></div></div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # Input area
    if prompt := st.chat_input("💬 Type your message here..."):
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "raw_content": prompt,
            "timestamp": timestamp
        })
        
        # Rerun to show user message immediately and trigger bot processing
        st.rerun()

    # If the last message is from the user, generate AI response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["raw_content"]
        
        # Show typing indicator
        with st.spinner("AI is typing..."):
            time.sleep(0.8) # Simulated processing time for animation feel
            result = bot.get_response(prompt)
            
            timestamp = datetime.datetime.now().strftime("%I:%M %p")
            
            # Format answer for HTML
            formatted_answer = result['answer'].replace('\\n', '<br>').replace('\n', '<br>')
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_answer,
                "raw_content": result['answer'],
                "confidence": result['confidence'],
                "timestamp": timestamp
            })
            
            st.rerun()

    # Footer
    st.markdown('<div class="footer">Powered by Python & Streamlit</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Check if running under Streamlit
    if runtime.exists():
        run_streamlit()
    else:
        parser = argparse.ArgumentParser(description="AI Chatbot Application")
        parser.add_argument(
            "--web",
            action="store_true",
            help="Run as a Flask web server instead of CLI",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=5000,
            help="Port for the web server (default: 5000)",
        )
        parser.add_argument(
            "--data",
            type=str,
            default="data.csv",
            help="Path to the Q&A dataset CSV file",
        )
        args = parser.parse_args()

        # Resolve data path relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, args.data)

        bot = Chatbot(data_path=data_path)

        if args.web:
            run_web(bot, port=args.port)
        else:
            run_cli(bot)
