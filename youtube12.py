from flask import Flask, request, jsonify,render_template
from test import *
from youtube2 import ad
import subprocess
app = Flask(__name__)

@app.route('/')
def receive_url34():
    return render_template("think12.html")

@app.route('/receive-url', methods=['POST'])
def receive_url():
    data = request.get_json()
    video_url = data.get('video_url')
    ad(video_url)

    if video_url:
        print(f"Received video URL: {video_url}")  
        return jsonify({"message": "Video URL received successfully!"}), 200
    else:
        return jsonify({"error": "No video URL received."}), 400

if __name__ == '__main__':
    app.run(debug=True,port=5005)
