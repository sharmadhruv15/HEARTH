"""
Hearth — Personalized Federated Assistant UI
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SHARED_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj"]
PERSONAL_MODULES = ["gate_proj", "up_proj", "down_proj"]

PERSONAS = {
    "mom": {
        "icon": "👩‍🍳",
        "label": "Mom",
        "desc": "Recipes & cooking",
        "color": "#e8845a",
        "weights": "C:/tmp/hearth_mom/personal_weights.pt",
        "placeholder": "Give me a recipe for chocolate chip cookies...",
    },
    "dad": {
        "icon": "💼",
        "label": "Dad",
        "desc": "Business emails",
        "color": "#5a8ee8",
        "weights": "C:/tmp/hearth_dad/personal_weights.pt",
        "placeholder": "Draft an email about the quarterly results...",
    },
    "you": {
        "icon": "💻",
        "label": "You",
        "desc": "Dev notes & ML",
        "color": "#5ae8a0",
        "weights": "C:/tmp/hearth_you/personal_weights.pt",
        "placeholder": "What did you learn about overfitting...",
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hearth — Federated Personal Assistant",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0f0f13; }
[data-testid="stSidebar"] { background: #141418; border-right: 1px solid #1e1e26; }

.hearth-header {
    padding: 0 0 24px 0;
    border-bottom: 1px solid #1e1e26;
    margin-bottom: 24px;
}
.hearth-title {
    font-size: 28px;
    font-weight: 600;
    color: #f0f0f5;
    letter-spacing: -0.5px;
    margin: 0;
}
.hearth-sub {
    font-size: 12px;
    color: #5a5a72;
    margin: 4px 0 0 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}

.persona-card {
    background: #18181f;
    border: 1px solid #22222e;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.persona-card:hover { border-color: #3a3a4e; }
.persona-card.active { border-color: var(--accent); background: #1c1c26; }
.persona-name { font-weight: 500; color: #d0d0e0; font-size: 14px; }
.persona-desc { font-size: 11px; color: #5a5a72; margin-top: 2px; }

.privacy-badge {
    background: #0d1f14;
    border: 1px solid #1a3d24;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 11px;
    color: #4a9e68;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 16px;
}

.chat-area {
    background: #0f0f13;
    min-height: 400px;
    padding: 8px 0;
}

.msg-user {
    background: #1a1a24;
    border: 1px solid #22222e;
    border-radius: 10px 10px 2px 10px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    color: #c0c0d0;
    font-size: 14px;
    line-height: 1.6;
}
.msg-assistant {
    background: #141420;
    border: 1px solid #1e1e2e;
    border-left: 3px solid var(--accent-color, #e8845a);
    border-radius: 2px 10px 10px 10px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    color: #d0d0e0;
    font-size: 14px;
    line-height: 1.7;
}
.msg-meta {
    font-size: 10px;
    color: #3a3a52;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

.privacy-log {
    background: #0a1a0a;
    border: 1px solid #1a3a1a;
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #3a7a3a;
    line-height: 1.8;
}

.stat-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}
.stat-box {
    background: #141418;
    border: 1px solid #1e1e26;
    border-radius: 8px;
    padding: 10px 12px;
    flex: 1;
    text-align: center;
}
.stat-val { font-size: 20px; font-weight: 600; color: #f0f0f5; }
.stat-key { font-size: 10px; color: #5a5a72; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

.stButton > button {
    background: #1e1e2a !important;
    color: #c0c0d0 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #252535 !important;
    border-color: #3a3a5a !important;
}

.stTextInput > div > div > input {
    background: #141418 !important;
    color: #d0d0e0 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

.stSelectbox > div > div {
    background: #141418 !important;
    border: 1px solid #2a2a3a !important;
    color: #d0d0e0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_base():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=SHARED_MODULES + PERSONAL_MODULES,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_cfg)
    return base, tok


def load_persona_model(client_name):
    model, tok = load_base()
    weights_path = PERSONAS[client_name]["weights"]
    if os.path.exists(weights_path):
        personal_sd = torch.load(weights_path, map_location="cpu")
        current_sd = get_peft_model_state_dict(model)
        matching = {k: v for k, v in personal_sd.items() if k in current_sd}
        current_sd.update(matching)
        set_peft_model_state_dict(model, current_sd)
    model.config.use_cache = True
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new_tokens=250):
    full_prompt = (
        f"<|system|>\nYou are a helpful personal assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    )
    inputs = tok(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    # strip prompt prefix from output
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()
    return decoded


# ── Session state ─────────────────────────────────────────────────────────────
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "mom"
if "histories" not in st.session_state:
    st.session_state.histories = {"mom": [], "dad": [], "you": []}
if "privacy_log" not in st.session_state:
    st.session_state.privacy_log = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "bytes_protected" not in st.session_state:
    st.session_state.bytes_protected = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="hearth-header">
        <p class="hearth-title">🔥 Hearth</p>
        <p class="hearth-sub">FEDERATED · PRIVATE · LOCAL</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Choose persona**")
    for key, p in PERSONAS.items():
        weights_exist = os.path.exists(p["weights"])
        status = "✓ trained" if weights_exist else "○ not trained"
        status_color = "#4a9e68" if weights_exist else "#5a5a72"
        if st.button(
            f"{p['icon']} {p['label']} — {p['desc']}",
            key=f"btn_{key}",
            use_container_width=True,
        ):
            st.session_state.active_persona = key
            st.rerun()

    st.markdown("---")

    # Privacy stats
    st.markdown("**Privacy monitor**")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-val">{st.session_state.total_queries}</div>
            <div class="stat-key">QUERIES</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">0</div>
            <div class="stat-key">DATA SENT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="privacy-badge">
        ✓ Raw data never left device<br>
        ✓ Only LoRA weights shared<br>
        ✓ Personal layers: local only<br>
        ✓ {len(st.session_state.privacy_log)} federated events logged
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Recent FL activity**")
    if st.session_state.privacy_log:
        log_text = "\n".join(st.session_state.privacy_log[-6:])
        st.markdown(f'<div class="privacy-log">{log_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="privacy-log">No activity yet.\nRun a query to begin.</div>', unsafe_allow_html=True)

    if st.button("Clear conversation", key="clear"):
        st.session_state.histories[st.session_state.active_persona] = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
persona = st.session_state.active_persona
p = PERSONAS[persona]

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"## {p['icon']} {p['label']}'s Assistant")
    weights_exist = os.path.exists(p["weights"])
    if weights_exist:
        st.markdown(f"<span style='color:#4a9e68;font-size:12px;font-family:monospace'>● Personal model loaded — {p['weights']}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#e85a5a;font-size:12px;font-family:monospace'>○ No personal weights found — run federated training first</span>", unsafe_allow_html=True)

st.markdown("---")

# Chat history
history = st.session_state.histories[persona]
if not history:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;color:#3a3a52;">
        <div style="font-size:40px;margin-bottom:12px">{p['icon']}</div>
        <div style="font-size:15px;font-weight:500;color:#5a5a72">{p['label']}'s personal assistant</div>
        <div style="font-size:12px;color:#3a3a52;margin-top:6px">Trained on private {p['desc'].lower()} data<br>Personal weights stay on this device</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-meta">YOU</div>
            <div class="msg-user">{msg["content"]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-meta">{p['icon']} {p['label'].upper()} · personal model · weights local</div>
            <div class="msg-assistant" style="border-left-color:{p['color']}">{msg["content"]}</div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Input
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            "Message",
            placeholder=p["placeholder"],
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send →", use_container_width=True)

if submitted and user_input.strip():
    if not os.path.exists(p["weights"]):
        st.error(f"No trained weights found for {p['label']}. Run `python run_clients.py` first to train the federated model.")
    else:
        history.append({"role": "user", "content": user_input})

        with st.spinner(f"Asking {p['label']}'s local model..."):
            model, tok = load_persona_model(persona)
            response = generate(model, tok, user_input)

        history.append({"role": "assistant", "content": response})

        # Update privacy stats
        st.session_state.total_queries += 1
        st.session_state.bytes_protected += len(user_input.encode())
        st.session_state.privacy_log.append(
            f"[query] {persona} · local inference · 0 bytes sent"
        )

        st.rerun()