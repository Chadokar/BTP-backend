import re
import ffmpeg
import datetime
from flask_cors import CORS
import speech_recognition as sr
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)


def convertToWAV(file):
    filename = f"{datetime.datetime.now().timestamp()}"
    inloc = f"uploads/{filename}.{re.split('[/;]',file.mimetype)[1]}"
    outloc = f"uploads/{filename}.wav"
    with open(inloc, "wb") as f:
        f.write(file.read())
    ffmpeg.input(inloc).output(outloc).run()
    return outloc


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "" or not file:
        return jsonify({"error": "No selected file"})

    if not re.match("audio/", file.mimetype):
        return jsonify({"error": "File type not supported"})
    if not re.match("audio/wav", file.mimetype):
        file = convertToWAV(file)

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file) as source:
            data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        return jsonify({"transcript": transcript})
    except Exception as e:
        print()
        print("error:", e)
        return jsonify({"error": "unable to transcribe"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
