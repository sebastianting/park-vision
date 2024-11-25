from flask import Flask, jsonify, Response, render_template
from parkingdetector import detect_parking_availability
import json

app = Flask(__name__)

def stream_parking_data(video_path='parking1slow.mp4', model_path='yolov8s.pt'):
    # Stream results frame by frame
    for availability in detect_parking_availability(video_path, model_path):
        yield f"data: {json.dumps(availability)}\n\n"

@app.route('/detect', methods=['GET'])
def detect_parking():
    return Response(stream_parking_data(video_path='parking1slow.mp4'), content_type='text/event-stream')

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/latest', methods=['GET'])
def get_latest_parking():
    global latest_parking_data
    if latest_parking_data:
        return jsonify({"status": "success", "data": latest_parking_data})
    else:
        return jsonify({"status": "error", "message": "No data available yet"})


if __name__ == '__main__':
    app.run(debug=True)
