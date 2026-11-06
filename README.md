# Hearth — Federated LoRA Personal Assistant

A Dockerized Federated Learning system for LLM fine-tuning where
multiple isolated clients collaboratively improve a shared language model
**without sharing their raw data.**

## Architecture

```
              Docker Network (hearth_net)

          +-------------------+
          |   Hearth Server   |
          | Aggregates ONLY   |
          | shared LoRA weights|
          +--------+----------+
                   ^
   shared weights  |  (attention layers only)
   ONLY            |
      +------------+------------+
      |            |            |
+-----+----+ +-----+----+ +----+-----+
| Mom      | | Dad      | | You      |
| Recipes  | | Emails   | | Dev notes|
| (private)| | (private)| | (private)|
+----------+ +----------+ +----------+

Raw text NEVER leaves client containers.
Personal weights (MLP layers) NEVER leave client containers.
```

## Shared vs Personal LoRA Layers

| Layer type | Modules | Role |
|---|---|---|
| Attention | q_proj, k_proj, v_proj, o_proj | **Shared** — aggregated by server |
| MLP | gate_proj, up_proj, down_proj | **Personal** — stays on each client |

## Setup

### 1. Copy your Phase 1 trained adapters and data into place

```
hearth/
├── data/
│   ├── mom/mom.jsonl        ← Mom's recipes data
│   ├── dad/dad.jsonl        ← Dad's emails data
│   └── you/you.jsonl        ← Your dev notes data
├── outputs/
│   ├── mom_adapter/         ← Unzip your downloaded mom_adapter here
│   ├── dad_adapter/         ← Unzip dad_adapter here
│   └── you_adapter/         ← Unzip you_adapter here
```

### 2. Local test (no Docker, before containerizing)

```bash
# Terminal 1 -- start the server
cd server
python app.py

# Terminal 2 -- start one client
cd client
CLIENT_NAME=mom SERVER_ADDR=localhost:8080 python client.py
```

### 3. Full Docker run

```bash
docker compose up --build
```

Watch the logs -- you will see:
- Server waiting for 3 clients
- Each client announcing it sends only shared weights
- Server printing aggregation stats (no raw data, only tensor shapes)
- Updated global shared weights broadcast back to clients

### 4. Verify privacy

```bash
# Inspect network traffic -- confirm only weight tensors cross the wire
docker logs hearth_server
docker logs hearth_client_mom
```

The server logs will show weight tensor shapes received.
No text strings, no raw data, no personal weights appear in server logs.

## Notes on 4GB VRAM (RTX 2050)

Phi-3-mini in 4-bit needs ~4GB VRAM. With a single 4GB card,
clients run **sequentially** (not in parallel) -- docker-compose
starts them one after another using `depends_on`. This is realistic
for a home-device federated scenario.

## Phase Roadmap

- [x] Phase 1 — Local LoRA fine-tuning (Colab notebooks)
- [ ] Phase 2 — Flower server + client (this folder)
- [ ] Phase 3 — Docker orchestration (docker-compose.yml ready)
- [ ] Phase 4 — Real non-IID data + personalization eval
- [ ] Phase 5 — Streamlit UI
