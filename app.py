from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import os
from bowling_sideview_analysis_v2 import process_video

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "🏏 Bowling Analyzer API is running."

@app.route('/process', methods=['POST'])
def process():
    try:
        video = request.files.get('video')
        arm = request.form.get('arm', '').lower()

        if not video or arm not in ['left', 'right']:
            return {"error": "Invalid video file or missing bowling arm (left/right)."}, 400

        # Save uploaded video to a temporary file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video.save(temp_input.name)

        # Call processing function
        output_path = process_video(temp_input.name, arm)

        return send_file(output_path, mimetype='video/mp4')

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
