from flask import Flask, request, send_file, render_template
import whisper
import os

app = Flask(__name__)

model = whisper.load_model("base")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Home ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Upload ----------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = model.transcribe(path)

    srt_path = "output.srt"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"]):
            start = format_time(seg["start"])
            end = format_time(seg["end"])

            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(seg["text"] + "\n\n")

    return send_file(srt_path, as_attachment=True)

# ---------------- Time format ----------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
