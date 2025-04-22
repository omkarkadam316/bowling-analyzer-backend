from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tempfile
import os
import logging
from werkzeug.utils import secure_filename
import time

# Import our bowling analysis function
from bowling_sideview_analysis_v2 import process_video

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Health check endpoint
@app.route('/')
def home():
    return "🏏 Bowling Analyzer Backend is Live!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is operational"}), 200

@app.route('/process', methods=['POST'])
def process():
    start_time = time.time()
    logger.info("Received video processing request")
    
    try:
        # Check if video file is in the request
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "No video file uploaded"}), 400
            
        video = request.files['video']
        
        # Check if arm selection is in the request
        if 'arm' not in request.form:
            logger.error("No arm selection in request")
            return jsonify({"error": "Please specify bowling arm (left/right)"}), 400
            
        arm = request.form['arm']
        
        if arm not in ['left', 'right']:
            logger.error(f"Invalid arm selection: {arm}")
            return jsonify({"error": "Arm must be either 'left' or 'right'"}), 400
        
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded video to temporary file
        input_filename = f"input_{int(time.time())}_{secure_filename(video.filename)}"
        temp_input_path = os.path.join(temp_dir, input_filename)
        video.save(temp_input_path)
        
        logger.info(f"Processing video saved to {temp_input_path}")
        logger.info(f"Processing with {arm} arm configuration")
        
        # Process the video
        try:
            output_path = process_video(temp_input_path, arm)
            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            
            # Send the processed video as response
            return send_file(output_path, mimetype='video/mp4', as_attachment=True,
                            download_name="bowling_analysis.mp4")
                            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp input file if it exists
        if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                logger.info(f"Cleaned up input file: {temp_input_path}")
            except:
                logger.warning(f"Could not remove temp input file: {temp_input_path}")

if __name__ == '__main__':
    # Use environment variables or defaults
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)