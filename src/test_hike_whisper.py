from datasets import load_dataset, Audio
import io
import soundfile as sf
import whisper


print("=" * 60)
print("HIKE KOREAN-ENGLISH WHISPER TEST")
print("=" * 60)

print("\nLoading HiKE...")

ds = load_dataset("thetaone-ai/HiKE")

# IMPORTANT:
# Prevent Hugging Face from automatically decoding audio with TorchCodec.
test = ds["test"].cast_column(
    "audio",
    Audio(decode=False)
)

print("Total clips:", len(test))
print("Columns:", test.column_names)


print("\nLoading Whisper Base...")

model = whisper.load_model("base")

print("Whisper loaded!")


print("\nRunning 5 samples...")
print("=" * 60)


for i in range(5):

    row = test[i]

    audio_info = row["audio"]

    print(f"\n[{i + 1}/5]")

    print("Audio path:", audio_info["path"])

    # Get raw audio bytes
    audio_bytes = audio_info["bytes"]

    # Decode the WAV audio ourselves
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    audio = audio.astype("float32")

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print("Sample rate:", sr)
    print("Audio length:", len(audio))

    result = model.transcribe(
        audio,
        task="transcribe"
    )

    prediction = result["text"].strip()

    detected_language = result.get(
        "language",
        "unknown"
    )

    print("Ground truth :", row["text"])
    print("Prediction    :", prediction)
    print("Language      :", detected_language)
    print("Category      :", row["category"])
    print("CS level      :", row["cs_level"])


print("\n" + "=" * 60)
print("5-SAMPLE TEST COMPLETE")
print("=" * 60)