"""Phase 2 diagnostic: verify exact matching behavior."""
import sys
sys.path.insert(0, ".")

from app.services.keyword_service import KeywordService, _normalize

ks = KeywordService()

# Check: what are the all_phrases normalized forms?
print("=== PHRASES THAT CONTAIN 'machine' ===")
for p in ks._all_phrases:
    if 'machine' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

print()
print("=== PHRASES THAT CONTAIN 'iot' ===")
for p in ks._all_phrases:
    if 'iot' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

print()
print("=== PHRASES THAT CONTAIN 'network' ===")
for p in ks._all_phrases:
    if 'network' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

print()
print("=== PHRASES THAT CONTAIN 'student' ===")
for p in ks._all_phrases:
    if 'student' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

print()
print("=== PHRASES THAT CONTAIN 'academic' ===")
for p in ks._all_phrases:
    if 'academic' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

print()
print("=== PHRASES THAT CONTAIN 'attendance' ===")
for p in ks._all_phrases:
    if 'attendance' in p.normalized:
        print(f"  original={repr(p.original)}, normalized={repr(p.normalized)}")

# Verify: "anomaly detection" matches because it IS a canonical term
print()
print("=== VERIFY: 'anomaly detection' in canonical index? ===")
print("  'anomaly detection' in canonical_index:", 'anomaly detection' in ks._canonical_index)

# Check what the match function actually does with the IoT text
print()
print("=== MATCH STEP-BY-STEP for IoT text ===")
text = "I want to develop a machine learning system that detects malicious network activity in IoT devices by analyzing network traffic and identifying abnormal behavior"
normalized = _normalize(text)
print(f"Normalized text: {repr(normalized)}")

# Check each phrase
matched_any = False
for p in ks._all_phrases:
    if p.normalized in normalized:
        print(f"  MATCH: {repr(p.original)} (normalized: {repr(p.normalized)})")
        matched_any = True

if not matched_any:
    print("  NO PHRASES MATCHED IN TEXT")

# Check: are there any 1-word phrases that match?
print()
print("=== SINGLE WORD CHECKS ===")
single_words = ['machine', 'learning', 'iot', 'network', 'traffic', 'abnormal', 'behavior',
                'student', 'academic', 'performance', 'attendance', 'examination', 'marks']
for w in single_words:
    in_canonical = w in ks._canonical_index
    in_synonym = w in ks._synonym_index
    in_alias = w in ks._alias_index
    in_abbr = w in ks._abbreviation_index
    in_phrases = any(p.normalized == w for p in ks._all_phrases)
    print(f"  '{w}': canonical={in_canonical}, synonym={in_synonym}, alias={in_alias}, abbr={in_abbr}, phrase={in_phrases}")
