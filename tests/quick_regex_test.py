import re

text = 'Reactive power is the imaginary component of the apparent power at the fundamental frequency, usually expressed in kilovars (kvar) [eurlex:631-28]. For contrast, active power is the real component [eurlex:631-20].'

# ✅ NEW PRIMARY PATTERN
bracket_pattern = r'\[\s*([a-zA-Z0-9]+:[a-zA-Z0-9\-_\.]+)\s*\]'
citations = re.findall(bracket_pattern, text, re.IGNORECASE)

print(citations)