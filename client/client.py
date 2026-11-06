"""
Hearth — Flower Federated Client
Trains a LoRA adapter locally, sends ONLY shared weights to the server.
Personal weights (MLP layers) never leave this machine.
Uses TinyLlama-1.1B (fits in 4GB VRAM with room for training).
"""

import os
import torch
import numpy as np
import flwr as fl
from typing import Dict, List
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    get_peft_model_state_dict, set_peft_model_state_dict
)
from datasets import load_dataset


# ── Config ────────────────────────────────────────────────────────────────────
CLIENT_NAME  = os.environ.get("CLIENT_NAME", "mom")
SERVER_ADDR  = os.environ.get("SERVER_ADDR", "localhost:8080")
DATA_PATH    = os.environ.get("DATA_PATH", f"../data/{CLIENT_NAME}/{CLIENT_NAME}.jsonl")
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Save personal weights to C:/tmp/ to avoid permission issues on outputs folder
PERSONAL_WEIGHTS_DIR = os.path.join("C:/tmp", f"hearth_{CLIENT_NAME}")
PERSONAL_WEIGHTS_PATH = os.path.join(PERSONAL_WEIGHTS_DIR, "personal_weights.pt")

SHARED_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj"]
PERSONAL_MODULES = ["gate_proj", "up_proj", "down_proj"]
MODULE_ROLE_MAP  = {name: "shared" for name in SHARED_MODULES}
MODULE_ROLE_MAP.update({name: "personal" for name in PERSONAL_MODULES})


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_adapter_state_dict(state_dict, module_role_map):
    shared, personal = {}, {}
    for key, val in state_dict.items():
        role = next((r for name, r in module_role_map.items() if name in key), None)
        if role == "shared":
            shared[key] = val
        elif role == "personal":
            personal[key] = val
        else:
            personal[key] = val
    return shared, personal


def state_dict_to_ndarrays(state_dict: Dict) -> List[np.ndarray]:
    return [v.cpu().float().numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(arrays: List[np.ndarray], keys: List[str]) -> Dict:
    return {k: torch.tensor(v) for k, v in zip(keys, arrays)}


def load_model_and_tokenizer():
    print(f"[{CLIENT_NAME}] Loading TinyLlama in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=SHARED_MODULES + PERSONAL_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_local_dataset(tokenizer):
    print(f"[{CLIENT_NAME}] Loading private dataset from {DATA_PATH}")
    raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def format_example(ex):
        text = (f"<|system|>\nYou are a helpful personal assistant.</s>\n"
                f"<|user|>\n{ex['prompt']}</s>\n"
                f"<|assistant|>\n{ex['response']}</s>")
        return {"text": text}

    formatted = raw_ds.map(format_example)

    def tokenize(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=512,
        )

    tokenized = formatted.map(tokenize, remove_columns=formatted.column_names)
    print(f"[{CLIENT_NAME}] Dataset ready: {len(tokenized)} examples")
    return tokenized


def load_personal_weights(model):
    if os.path.exists(PERSONAL_WEIGHTS_PATH):
        print(f"[{CLIENT_NAME}] Restoring personal weights from {PERSONAL_WEIGHTS_PATH}")
        personal_sd = torch.load(PERSONAL_WEIGHTS_PATH, map_location="cpu")
        current_sd = get_peft_model_state_dict(model)
        matching = {k: v for k, v in personal_sd.items() if k in current_sd}
        current_sd.update(matching)
        set_peft_model_state_dict(model, current_sd)
        print(f"[{CLIENT_NAME}] Loaded {len(matching)} matching personal weight tensors.")
    else:
        print(f"[{CLIENT_NAME}] No existing personal weights — starting fresh.")


def save_personal_weights(model):
    os.makedirs(PERSONAL_WEIGHTS_DIR, exist_ok=True)
    sd = get_peft_model_state_dict(model)
    _, personal_sd = split_adapter_state_dict(sd, MODULE_ROLE_MAP)
    torch.save(personal_sd, PERSONAL_WEIGHTS_PATH)
    print(f"[{CLIENT_NAME}] Personal weights saved to {PERSONAL_WEIGHTS_PATH} (not transmitted).")


# ── Flower Client ─────────────────────────────────────────────────────────────
class HearthClient(fl.client.NumPyClient):

    def __init__(self):
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.dataset = load_local_dataset(self.tokenizer)
        self.shared_keys = None
        load_personal_weights(self.model)

    def get_parameters(self, config):
        sd = get_peft_model_state_dict(self.model)
        shared_sd, _ = split_adapter_state_dict(sd, MODULE_ROLE_MAP)
        self.shared_keys = list(shared_sd.keys())
        arrays = state_dict_to_ndarrays(shared_sd)
        print(f"[{CLIENT_NAME}] Sending {len(arrays)} shared weight tensors. "
              f"Personal weights withheld.")
        return arrays

    def set_parameters(self, parameters):
        if self.shared_keys is None:
            sd = get_peft_model_state_dict(self.model)
            shared_sd, _ = split_adapter_state_dict(sd, MODULE_ROLE_MAP)
            self.shared_keys = list(shared_sd.keys())
        new_shared_sd = ndarrays_to_state_dict(parameters, self.shared_keys)
        current_sd = get_peft_model_state_dict(self.model)
        current_sd.update(new_shared_sd)
        set_peft_model_state_dict(self.model, current_sd)
        print(f"[{CLIENT_NAME}] Global shared weights applied. Personal weights unchanged.")

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=os.path.join("C:/tmp", f"hearth_train_{CLIENT_NAME}"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="no",
            fp16=True,
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
        )

        print(f"[{CLIENT_NAME}] Starting local training on private data...")
        print(f"[{CLIENT_NAME}] Raw data stays here — only adapter weights leave.")
        trainer.train()
        save_personal_weights(self.model)
        return self.get_parameters(config={}), len(self.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.dataset), {}


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    print(f"\n[{CLIENT_NAME}] Hearth Federated Client starting...")
    print(f"[{CLIENT_NAME}] Connecting to server at {SERVER_ADDR}")
    print(f"[{CLIENT_NAME}] Raw data NEVER leaves this machine.\n")

    fl.client.start_numpy_client(
        server_address=SERVER_ADDR,
        client=HearthClient(),
    )


if __name__ == "__main__":
    main()
