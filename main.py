from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from pydub import AudioSegment
from io import BytesIO
import traceback
import tempfile
import math

app = Flask(__name__)
CORS(app)
model = whisper.load_model("base")


def split_audio(audio_segment, chunk_length_ms=30000):
    """
    Splits the audio segment into smaller chunks of given length in milliseconds.
    """
    length_ms = len(audio_segment)
    return [
        audio_segment[i : i + chunk_length_ms]
        for i in range(0, length_ms, chunk_length_ms)
    ]


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        audio_segment = AudioSegment.from_file(BytesIO(file.read()))
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_sample_width(2)

        # Split the audio into chunks
        chunks = split_audio(audio_segment)

        transcriptions = []
        for chunk in chunks:
            # Save the chunk to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk.export(temp_file.name, format="wav")

                # Load chunk for Whisper and transcribe
                audio_data = whisper.load_audio(temp_file.name)
                audio_data = whisper.pad_or_trim(audio_data)

                result = model.transcribe(audio_data)
                chunk_transcription = result[
                    "text"
                ].strip()  # Trim leading/trailing spaces
                transcriptions.append(chunk_transcription)

        full_transcription = " ".join(transcriptions).strip()  # Ensure no leading space
        print(
            "Transcription:", full_transcription
        )  # Print full transcription to the server console

        return jsonify({"transcription": full_transcription})
    except Exception as e:
        app.logger.error("Error in transcribe_audio: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
