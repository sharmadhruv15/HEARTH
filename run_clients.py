"""
Hearth — Sequential Client Runner with auto server restart.
Starts a fresh server for each client, runs one federated round each.
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

clients = [
    {"CLIENT_NAME": "mom", "DATA_PATH": os.path.join(BASE_DIR, "data", "mom", "mom.jsonl")},
    {"CLIENT_NAME": "dad", "DATA_PATH": os.path.join(BASE_DIR, "data", "dad", "dad.jsonl")},
    {"CLIENT_NAME": "you", "DATA_PATH": os.path.join(BASE_DIR, "data", "you", "you.jsonl")},
]

print("=" * 60)
print("  Hearth — Sequential Federated Client Runner")
print("  Clients: mom -> dad -> you")
print("  Server restarts between each client.")
print("  Personal weights saved to C:/tmp/ (never transmitted).")
print("=" * 60)

for i, client in enumerate(clients):
    print(f"\n[Runner] Starting server for round {i+1}/3...")
    server_proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "server", "app.py")],
        cwd=BASE_DIR,
    )
    time.sleep(4)  # wait for server to boot

    env = os.environ.copy()
    env["CLIENT_NAME"] = client["CLIENT_NAME"]
    env["SERVER_ADDR"] = "localhost:8080"
    env["DATA_PATH"] = client["DATA_PATH"]

    print(f"[Runner] Starting client {i+1}/3: {client['CLIENT_NAME']}")
    print(f"[Runner] Personal weights: C:/tmp/hearth_{client['CLIENT_NAME']}/personal_weights.pt")
    print(f"[Runner] {'='*50}\n")

    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "client", "client.py")],
        env=env,
        cwd=BASE_DIR,
    )

    server_proc.terminate()
    server_proc.wait()
    time.sleep(2)  # let port 8080 release

    if result.returncode != 0:
        print(f"\n[Runner] WARNING: {client['CLIENT_NAME']} exited with code {result.returncode}")
    else:
        print(f"\n[Runner] {client['CLIENT_NAME']} completed successfully.")

print("\n" + "=" * 60)
print("  All 3 clients finished.")
print("  Personal weights saved under C:/tmp/hearth_<client>/")
print("=" * 60)