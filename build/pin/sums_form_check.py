"""Read-only: score_ecs.provisional_items on <ECS> as it stands (no APPROVED yet), to show the scorer accepts
SHA256SUMS.txt's form and finds every listed file at its sha256.  Writes nothing."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/tools")
import score_ecs
items = score_ecs.provisional_items("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs", [])
for x in items:
    print("ITEM:", x)
bad = [x for x in items if "SHA256SUMS" in x or "pinned file" in x or "M1/M5" in x or "code copy" in x]
print("SUMS/PIN/SEAL ITEMS:", len(bad), "(expected 0); other items:", len(items) - len(bad), "(expected: no APPROVED, no criteria.txt)")
