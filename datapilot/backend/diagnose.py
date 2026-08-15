"""Diagnostic script to find root cause of dictionary matching failure."""
import sys
sys.path.insert(0, ".")

from app.services.keyword_service import KeywordService

ks = KeywordService()

# Test 1: What does the dictionary actually contain?
print("=== DICTIONARY CONTENTS ===")
print(f"Canonical index entries: {len(ks._canonical_index)}")
print(f"Synonym index entries: {len(ks._synonym_index)}")
print(f"Alias index entries: {len(ks._alias_index)}")
print(f"Abbreviation index entries: {len(ks._abbreviation_index)}")
print(f"All phrases: {len(ks._all_phrases)}")
print()

# Show all canonical terms
print("=== ALL CANONICAL TERMS ===")
for term in sorted(ks._canonical_index.keys()):
    print(f"  {repr(term)}")
print()

# Show sample synonyms
print("=== SAMPLE SYNONYMS (first 30) ===")
count = 0
for key, entries in ks._synonym_index.items():
    if count >= 30:
        break
    print(f"  {repr(key)} -> {entries[0]['canonical_term']}")
    count += 1
print()

# Show sample all_phrases
print("=== SAMPLE ALL_PHRASES (first 30) ===")
for p in ks._all_phrases[:30]:
    print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")
print()

# Test 2: What happens with the IoT test input?
print("=== MATCHING: IoT test ===")
text = "I want to develop a machine learning system that detects malicious network activity in IoT devices by analyzing network traffic and identifying abnormal behavior"
matches = ks.match(text)
print(f"Matches found: {len(matches)}")
for m in matches:
    print(f"  {m.canonical_term} (type={m.match_type.value}, text={m.matched_text})")
print()

# Test 3: Direct phrase searches
print("=== DIRECT PHRASE SEARCHES ===")
phrases = ["machine learning", "iot", "IoT", "network traffic", "anomaly detection", "phishing", "classification", "student", "academic performance", "attendance"]
for phrase in phrases:
    matches = ks.match(phrase)
    print(f"  \"{phrase}\" -> {len(matches)} matches: {[m.canonical_term for m in matches]}")
print()

# Test 4: Check what normalization does
print("=== NORMALIZATION CHECK ===")
from app.services.keyword_service import _normalize
test_texts = [
    "machine learning",
    "Machine Learning",
    "MACHINE LEARNING",
    "network traffic",
    "network-traffic",
    "IoT",
    "iot",
    "abnormal behavior",
    "abnormal behaviour",
]
for t in test_texts:
    print(f"  {repr(t)} -> {repr(_normalize(t))}")
