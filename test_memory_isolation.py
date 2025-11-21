"""
Test memory isolation between regions using the same logic as memory_agent.py.
Run: python test_memory_isolation.py
"""

import os
import shutil
from agents.memory_agent import (
    store_interaction,
    recall_similar,
    delete_memories_for_region,
    list_all_regions
)

def reset_store():
    """Nuclear reset: wipe memory_store directory so we start clean."""
    if os.path.isdir("memory_store"):
        shutil.rmtree("memory_store")
    print("🔥 Reset: memory_store wiped.")

def assert_only_regions(expected):
    regions = list_all_regions()
    print(f"📂 Regions present: {regions}")
    assert set(regions) == set(expected), f"Expected regions {expected}, got {regions}"

def assert_contains_only(region, phrases, mode=None):
    """Check that recall for region returns only the given phrases."""
    results = recall_similar(region, "", mode=mode)  # show all for region
    got_phrases = [m.get("phrase", "") if isinstance(m, dict) else m for m in results]
    print(f"🔎 {region} memories: {got_phrases}")
    assert set(got_phrases) == set(phrases), f"{region} should have {phrases}, got {got_phrases}"

def main():
    reset_store()

    # 1) Store New York memories
    ny_region = "United States (New York)"
    store_interaction(ny_region, "Where is the subway?", tone="Neutral", gesture="Polite nod", custom="Use 'subway' not 'metro'", mode="Text")
    store_interaction(ny_region, "Thank you!", tone="Warm", gesture="Smile", custom="Direct, friendly", mode="Text")

    # 2) Store Chennai memories
    ch_region = "Chennai"
    store_interaction(ch_region, "Bus stop enga iruku?", tone="Curious", gesture="Open palm", custom="Use Tamil if possible", mode="Text")
    store_interaction(ch_region, "Nandri!", tone="Warm", gesture="Smile", custom="Tamil for thank you", mode="Text")

    # 3) Verify region list isolation
    assert_only_regions([ny_region, ch_region])

    # 4) Verify recall isolation (show all for each region)
    assert_contains_only(ny_region, ["Where is the subway?", "Thank you!"], mode=None)
    assert_contains_only(ch_region, ["Bus stop enga iruku?", "Nandri!"], mode=None)

    # 5) Cross-check: similarity queries should not bleed across regions
    ny_sim = recall_similar(ny_region, "subway", mode=None)
    ny_sim_phrases = [m.get("phrase", "") if isinstance(m, dict) else m for m in ny_sim]
    print(f"🧪 Similar in NY for 'subway': {ny_sim_phrases}")
    assert any("subway" in p.lower() for p in ny_sim_phrases), "NY similarity should retrieve the subway phrase."
    assert all("enga" not in p.lower() and "nandri" not in p.lower() for p in ny_sim_phrases), "NY results should not include Chennai phrases."

    ch_sim = recall_similar(ch_region, "enga", mode=None)
    ch_sim_phrases = [m.get("phrase", "") if isinstance(m, dict) else m for m in ch_sim]
    print(f"🧪 Similar in Chennai for 'enga': {ch_sim_phrases}")
    assert any("enga" in p.lower() for p in ch_sim_phrases), "Chennai similarity should retrieve the 'enga' phrase."
    assert all("subway" not in p.lower() and "thank you" not in p.lower() for p in ch_sim_phrases), "Chennai results should not include New York phrases."

    # 6) Delete Chennai and re-verify isolation
    print("🧹 Deleting Chennai memories...")
    msg = delete_memories_for_region(ch_region)
    print(msg)
    assert_only_regions([ny_region])
    assert_contains_only(ny_region, ["Where is the subway?", "Thank you!"], mode=None)

    print("✅ All isolation tests passed.")

if __name__ == "__main__":
    main()
