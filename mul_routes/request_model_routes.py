from PIL import Image
import json, os, io
from flask import Blueprint, jsonify, request
import google.generativeai as genai
from dotenv import load_dotenv
from googletrans import Translator
import pickle
import numpy as np
import time
import requests

load_dotenv()

request_model_bp = Blueprint('request_model_bp', __name__)
genai.configure(api_key=os.getenv("MODEL"))
UPLOAD_API_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_api_data/')

def validate_api_key(api_key):
    """Loads existing API key data from api.json and validates the provided API key."""
    try:
        with open(f'{UPLOAD_API_FOLDER}api.json', 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        return False  # Return False if the file does not exist

    # Check if the provided API key exists in the loaded data
    for entry in existing_data:
        if entry['generated_api_key'] == api_key:
            return True  # Valid API key

    return False  # Invalid API key

def process_iming(role, prompt):
    """Processes the prompt using the generative model."""
    full_prompt = role + "\n" + prompt
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(full_prompt)
        
        return response.text
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while generating content: {e}"

def process_analyze_image(role, image, prompt):
    """Processes the image and prompt using the generative vision model."""
    
    defaultRole = "You are an advanced image analysis model tasked with examining and providing detailed insights about images. Your role is to accurately interpret the visual content and provide a comprehensive description or analysis based on the provided image:"
    full_prompt = defaultRole + role + ". " + prompt
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Ensure that the image and prompt are correctly passed to the API
        response = model.generate_content([image, full_prompt])
        
        return response.text
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while processing the image: {e}"
    
@request_model_bp.route('/request_iming', methods=['POST'])
def request_iming():
    if request.is_json:
        data = request.get_json()
        api_key = data.get('api_key')
        prompt = data.get('prompt', "Say Hi!")
        role = data.get('role', "'Mul-Model' AI Chatbot Bot")
    else:
        api_key = request.form.get('api_key')
        prompt = request.form.get('prompt', "Say Hi!") 
        role = request.form.get('role', "'Mul-Model' AI Chatbot Bot")
    
    # Validate API key
    if not validate_api_key(api_key):
        return jsonify({"error": "Invalid API key"}), 400

    # Process the prompt
    text_by_iming = process_iming(role, prompt)
    result = {"text": text_by_iming}
    return jsonify(result), 200


@request_model_bp.route('/request_analyze_image', methods=['POST'])
def request_analyze_image():
    api_key = request.form.get('api_key')
    prompt = request.form.get('prompt', "What can you see in this image?")
    model_type = request.form.get('model_type')
    
    if model_type == "captain" :
        role = "Giving Captions to the image: "
        prompt = "Give a caption for this image to upload in my social media. Generate 5 captions in points in '\n' after each point and also provide respective emojis."
    elif model_type == "vision":
        role = "Analyze and Explain Details in image: "
    elif model_type == "xvc":
        role = "Imagine you are a xray analyse and trying to analyze this given xray image: "
    else:
        return jsonify({"error": "Type of model is not mentioned in request"}), 400
        
    # Validate API key
    if not validate_api_key(api_key):
        return jsonify({"error": "Invalid API key"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Process the image and prompt using the Gemini API
        text_by_vision = process_analyze_image(role, image, prompt)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = {"text": text_by_vision}
    return jsonify(result), 200


@request_model_bp.route('/request_analyze_harmony', methods=['POST'])
def request_analyze_harmony():
    translator = Translator()
    api_key = request.form.get('api_key')

    # Validate API key
    if not validate_api_key(api_key):
        return jsonify({"error": "Invalid API key"}), 400

    payload = request.form.get('payload', None)

    if payload is None:
        return jsonify({"error": "No payload provided"}), 400

    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid payload format"}), 400

    model_type = request.form.get('model_type', "translator")
    if model_type == "translator":
        text = payload_data.get('text', '')
        target_lang = payload_data.get('targetLang', '')

        if not text or not target_lang:
            return jsonify({"error": "Text and target language must be provided"}), 400
        
        translated = translator.translate(text, dest=target_lang)
        result = {"text": translated.text}
    else:
        return jsonify({"error": "Unsupported model type"}), 400

    return jsonify(result), 200

@request_model_bp.route('/request_generate_image', methods=['POST'])
def request_generate_image():
    """
    Sends a request to the Hugging Face API to generate an image based on the provided input 
    and saves the generated image in a designated upload directory.
    """
    api_key = request.form.get('api_key')
    prompt = request.form.get('prompt')

    # Validate API key
    if not validate_api_key(api_key):
        return jsonify({"error": "Invalid API key"}), 400
    
    # Validate prompt
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}", 
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt}

    try:
        # Send a POST request to the Hugging Face API
        response = requests.post(url, headers=headers, json=payload, stream=True)
        if response.status_code != 200:
            return jsonify({"error": f"API request failed with status {response.status_code}: {response.text}"}), 500

        # Save the generated image
        output_directory = os.path.join(os.getcwd(), 'mul_routes/store_gen_img/')
        os.makedirs(output_directory, exist_ok=True)

        # Use a more appropriate filename
        filename = f"{int(time.time())}.jpg"
        
        output_path = os.path.join(output_directory, filename)
        with open(output_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=8192):
                output_file.write(chunk)

        base_url = request.host_url
        file_path = f"{base_url}get_file/gen_img/{filename}"

        # Return the file path as the get from send_files_routes
        return jsonify({"file_path": file_path}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# new
@request_model_bp.route('/trigger_custom_model', methods=['POST'])
def request_custom_model():
    model_input = request.form.get('model_input')
    model_path = request.form.get('model_path')

    if not model_input or not model_path:
        return jsonify({"output": "Both 'model_input' and 'model_path' are required."}), 400

    model_file = os.path.join(UPLOAD_API_FOLDER, "models", model_path)
    if not os.path.exists(model_file):
        return jsonify({"output": f"Model file '{model_file}' not found."}), 404
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        parsed_input = [[float(x) for x in sublist] for sublist in eval(model_input)]
        predictions = model.predict(parsed_input)
        print(predictions)
        return jsonify({"output": str(predictions)}), 200

    except Exception as e:
        return jsonify({"output": f"Error during prediction: {str(e)}"}), 200