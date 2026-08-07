import os
import sqlite3
import json
from dotenv import load_dotenv

# load env manually
load_dotenv()

from memorymesh.main import store_memory, MemoryRequest
from memorymesh.memory import DB_FILE, _init_local_db, get_all_nodes, get_all_edges

def main():
    # Setup test by clearing the local db
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    _init_local_db()

    # Step 1: Insert first mention
    print("Ingesting Fact 1...")
    req1 = MemoryRequest(text="William Henry Gates III founded Microsoft in 1975.")
    res1 = store_memory(req1)
    print("Fact 1 Response:", res1)
    
    # Check nodes
    nodes = get_all_nodes()
    print("Nodes after Fact 1:")
    for n in nodes:
        print(f" - [{n['id']}] {n['content']} (canonical: {n.get('canonical_name')}, aliases: {n.get('aliases')})")
    
    # Step 2: Insert second mention (alias)
    print("\nIngesting Fact 2...")
    req2 = MemoryRequest(text="Bill Gates stepped down as Microsoft board chairman in 2014.")
    res2 = store_memory(req2)
    print("Fact 2 Response:", res2)

    # Check nodes again
    nodes = get_all_nodes()
    print("\nNodes after Fact 2:")
    for n in nodes:
        print(f" - [{n['id']}] {n['content']} (canonical: {n.get('canonical_name')}, aliases: {n.get('aliases')})")

    # The count of nodes containing 'Gates' or 'William' should be 1 if merged, >1 if not merged.
    gates_nodes = [n for n in nodes if "Gates" in n['content'] or "William" in n['content'] or (n.get('canonical_name') and "Gates" in n.get('canonical_name'))]
    print(f"\nNumber of 'Gates' entity nodes: {len(gates_nodes)} (Expected 1 if MERGE worked)")
    if len(gates_nodes) == 1:
        print("MERGE SUCCESS!")
    else:
        print("MERGE FAILED: Multiple nodes found for the same entity.")

if __name__ == "__main__":
    # temporarily clear SUPABASE vars to force local fallback
    if "SUPABASE_URL" in os.environ:
        del os.environ["SUPABASE_URL"]
    if "SUPABASE_KEY" in os.environ:
        del os.environ["SUPABASE_KEY"]
    main()
