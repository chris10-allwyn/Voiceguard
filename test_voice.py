from speaker_verify import compare_voice

audio = "recordings/charvi1.wav.ogg"

similarity = compare_voice(audio)

print("\n==============================")
print("VOICE VERIFICATION RESULT")
print("==============================")
print("Similarity:", round(similarity, 3))

if similarity >= 0.75:
    print("✅ MATCH — Registered User")
else:
    print("❌ NOT MATCH — Unknown Voice")