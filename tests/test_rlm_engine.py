import pytest
import time
from backend.rlm.engine import RLMEngine, RLMConfig
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_capability_needle_in_haystack():
    # Construct a massive input (1 million chars)
    haystack_chunk = "The weather is nice today. " * 5000
    needle = "SECRET_CODE_994827 is the password."
    haystack = haystack_chunk + needle + haystack_chunk
    
    query = "Find the password that looks like SECRET_CODE_XXXXXX. Use a python script to search the CONTEXT_DATA using regex."
    
    start = time.time()
    response = client.post("/rlm/query", json={
        "context": haystack,
        "query": query
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "SECRET_CODE_994827" in data["answer"]
    
    stats = data["stats"]
    assert stats["elapsed_seconds"] > 0
    print(f"\n[Capability Test] Found needle in {stats['elapsed_seconds']}s")
    print(f"[Capability Test] Tokens used: {stats['tokens_used']}")
    print(f"[Capability Test] Subcalls made: {stats['subcalls_made']}")

def test_safety_cap_wall_clock():
    # Force a timeout by instructing the model to write an infinite loop, or test the sandbox timeout directly.
    # We will test the sandbox timeout directly to ensure it doesn't hang.
    engine = RLMEngine(RLMConfig(max_wall_clock_seconds=2.0))
    infinite_loop_code = "while True: pass"
    
    start = time.time()
    output = engine._run_sandbox(infinite_loop_code, "test context", 1)
    elapsed = time.time() - start
    
    assert "Capped: Safety limits exceeded" in output
    assert elapsed < 3.0
    print(f"\n[Safety Cap Test] Blocked infinite loop in {elapsed:.2f}s")

def test_sandbox_boundary():
    engine = RLMEngine()
    
    # Attempt 1: Network access
    malicious_code_network = """
try:
    import urllib.request
    urllib.request.urlopen("http://example.com")
    print("NETWORK_SUCCESS")
except Exception as e:
    print(f"NETWORK_BLOCKED: {type(e).__name__}")
"""
    output1 = engine._run_sandbox(malicious_code_network, "test context", 1)
    
    assert "NETWORK_SUCCESS" not in output1
    assert "NETWORK_BLOCKED" in output1 or "NoneType" in output1 or "KeyError" in output1 or "ModuleNotFoundError" in output1
    print(f"\n[Sandbox Boundary] Network blocked: {output1.strip()}")
    
    # Attempt 2: Filesystem access
    malicious_code_fs = """
try:
    with open("secret.txt", "w") as f:
        f.write("hacked")
    print("FS_SUCCESS")
except Exception as e:
    print(f"FS_BLOCKED: {type(e).__name__}")
"""
    output2 = engine._run_sandbox(malicious_code_fs, "test context", 1)
    
    assert "FS_SUCCESS" not in output2
    assert "FS_BLOCKED: NameError" in output2 or "open is not defined" in output2 or "NameError" in output2
    print(f"[Sandbox Boundary] Filesystem blocked: {output2.strip()}")
