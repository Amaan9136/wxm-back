from flask import Blueprint, send_file, jsonify, request, render_template
from pyngrok import ngrok
import socket
import os
import pickle
import numpy as np
import ast  

live_inference_bp = Blueprint('live_inference_bp', __name__)
shared_directory = None

# Function to get the local IP address
def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# Start Ngrok and return public URL
def start_ngrok():
    # Ensure ngrok tunnel is started only once
    if not ngrok.get_tunnels():
        public_url = ngrok.connect(5000).public_url
    else:
        public_url = ngrok.get_tunnels()[0].public_url
    return public_url

# Endpoint to retrieve connection details and list of model names
# local = socket
@live_inference_bp.route('/local-details', methods=['GET'])
def get_local_details():
    local_ip = get_local_ip()

    return jsonify({ 
        "local_ip": f"http://{local_ip}:5000/",
        "local_files": f"http://{local_ip}:5000/files",
        "local_read": f"http://{local_ip}:5000/read/(filename)",
        "local_download": f"http://{local_ip}:5000/download/(filename)",
        "local_predict": f"http://{local_ip}:5000/predict/(filename)"
    })

# global = ngrok
@live_inference_bp.route('/global-details', methods=['GET'])
def get_global_details():
    ngrok_url = start_ngrok()
    
    return jsonify({
        "ngrok_url": ngrok_url,
        "ngrok_files": ngrok_url+"/files",
        "ngrok_read": ngrok_url+"/read/(filename)",
        "ngrok_download": ngrok_url+"/download/(filename)",
        "ngrok_predict": ngrok_url+"/predict/(filename)",
    })

# Endpoint for participants to download files from the shared directory
@live_inference_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if not shared_directory:
        return jsonify({"error": "No shared directory set by the host."}), 400
    
    file_path = os.path.join(shared_directory, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404
    
# Endpoint to list files in the shared directory for participants
@live_inference_bp.route('/files', methods=['GET'])
def list_files():
    if not shared_directory:
        return jsonify({"error": "No shared directory set by the host."}), 400
    
    try:
        # List files in the shared directory
        file_list = os.listdir(shared_directory)
        return jsonify({"files": file_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Endpoint to set the shared directory path from the host
@live_inference_bp.route('/submit-path', methods=['POST'])
def submit_path():
    global shared_directory
    shared_directory = request.form.get('path')

    # Check if a path is provided
    if not shared_directory:
        return jsonify({"error": "No shared directory set by the host."}), 400
    
    # Check if the directory exists
    if not os.path.exists(shared_directory):
        return jsonify({"error": "Path does not exist on the server."}), 400
    
    try:
        # List files in the shared directory
        file_list = os.listdir(shared_directory)
        return jsonify({"message": "Path set successfully!", "files": file_list, "shared_directory": shared_directory})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Endpoint to handle prediction requests using query parameters
@live_inference_bp.route('/predict/<model_name>', methods=['GET'])
def predict(model_name):
    file_path = os.path.join(shared_directory, model_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "Model file not found"}), 404

    # Load the model
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    # Get input data from query parameters as a string
    input_data_str = request.args.get('data', type=str)
    if not input_data_str:
        return jsonify({"error": "No input data provided"}), 400

    # Safely evaluate the string to a Python literal (like a list, tuple, etc.)
    try:
        input_data = ast.literal_eval(input_data_str)
    except Exception as e:
        return jsonify({"error": f"Failed to parse input data: {str(e)}"}), 400

    # Ensure the input data is in the correct format for prediction
    input_array = np.array(input_data)

    # Make a prediction
    try:
        prediction = model.predict(input_array)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the prediction result
    return jsonify({"prediction": prediction.tolist()})

# New endpoint to read data from `.txt` or `.md` files
@live_inference_bp.route('/read/<filename>', methods=['GET'])
def read_model_data(filename):
    # # Allow only '.txt' or files with '.md' extension
    if not (filename.endswith(".txt") or filename.endswith(".md")):
        return jsonify({"error": "Access denied: only '.txt' or '.md' files can be read"}), 403

    file_path = os.path.join(shared_directory, filename)
    print(file_path)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, 'r') as f:
            data = f.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read data: {str(e)}"}), 500

    return jsonify({
        "data": data,
    })