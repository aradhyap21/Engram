import os
import json
from dotenv import load_dotenv

# load env manually
load_dotenv()

from backend.main import store_memory, MemoryRequest
from backend.memory import DB_FILE, _init_local_db, get_all_edges, get_active_edges, get_all_nodes

def main():
    # Setup test by clearing the local db
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    _init_local_db()

    # Step 1: Insert first mention
    print("Ingesting Fact 1...")
    req1 = MemoryRequest(text="Sarah Chen lives in Boston (2021).")
    res1 = store_memory(req1)
    print("Fact 1 Response:", res1)
    
    # Step 2: Insert second mention
    print("\nIngesting Fact 2...")
    req2 = MemoryRequest(text="Sarah Chen has been living in Seattle since March 2024.")
    res2 = store_memory(req2)
    print("Fact 2 Response:", res2)

    # Check nodes to find Sarah Chen's ID
    nodes = get_all_nodes()
    sarah_id = None
    for n in nodes:
        if "Sarah Chen" in n['content']:
            sarah_id = n['id']
            break

    if not sarah_id:
        print("Failed to find Sarah Chen node.")
        return

    # Check edges
    all_edges = get_all_edges()
    print("\nAll edges after both facts:")
    for e in all_edges:
        print(f" - [{e['id']}] {e['relationship']}: valid_at={e.get('valid_at')}, invalid_at={e.get('invalid_at')}, fact='{e.get('fact_text')}'")
    
    # Check active edges specifically for lives_in
    active_edges = get_active_edges(sarah_id, "lives_in")
    print(f"\nActive 'lives_in' edges for Sarah Chen ({len(active_edges)} found):")
    for e in active_edges:
        print(f" - [{e['id']}] {e['relationship']}: fact='{e.get('fact_text')}'")

    # Verification
    boston_edge = next((e for e in all_edges if e.get('fact_text') and 'Boston' in e['fact_text']), None)
    seattle_edge = next((e for e in all_edges if e.get('fact_text') and 'Seattle' in e['fact_text']), None)

    success = True
    if boston_edge:
        if boston_edge.get("invalid_at"):
            print("SUCCESS: Boston edge has invalid_at set:", boston_edge["invalid_at"])
        else:
            print("FAILED: Boston edge was not invalidated.")
            success = False
    else:
        print("FAILED: Boston edge not found. Was it deleted?")
        success = False

    if seattle_edge:
        if not seattle_edge.get("invalid_at"):
            print("SUCCESS: Seattle edge is currently active.")
        else:
            print("FAILED: Seattle edge should be active but has invalid_at.")
            success = False
    else:
        print("FAILED: Seattle edge not found.")
        success = False

    active_facts = [e.get("fact_text", "") for e in active_edges]
    if any("Seattle" in fact for fact in active_facts) and not any("Boston" in fact for fact in active_facts):
        print("SUCCESS: get_active_edges() returns only Seattle.")
    else:
        print("FAILED: get_active_edges() did not return exactly the correct edges.")
        success = False

    if success:
        print("\nALL CONFLICT DETECTION TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED.")

if __name__ == "__main__":
    # temporarily clear SUPABASE vars to force local fallback
    if "SUPABASE_URL" in os.environ:
        del os.environ["SUPABASE_URL"]
    if "SUPABASE_KEY" in os.environ:
        del os.environ["SUPABASE_KEY"]
    main()
