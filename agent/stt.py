from faster_whisper import WhisperModel

_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> str:
    segments, _info = _MODEL.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python agent/stt.py <audio_file>")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print("Transcript:", result)

