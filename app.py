from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
from bowling_sideview_analysis_v2 import process_video

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return "🏏 Bowling Analyzer Backend is Live!"

@app.route('/process', methods=['POST'])
def process():
    try:
        video = request.files['video']
        arm = request.form['arm']

        # Save uploaded video to temporary file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video.save(temp_input.name)

        # Process the video using custom function
        output_path = process_video(temp_input.name, arm)

        # Send the processed video as response
        return send_file(output_path, mimetype='video/mp4')

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
