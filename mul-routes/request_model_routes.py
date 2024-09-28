from PIL import Image
import json, os, re, io
from flask import Blueprint, jsonify, request
import google.generativeai as genai
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()

request_model_bp = Blueprint('request_model_bp', __name__)
genai.configure(api_key=os.getenv("MODEL"))
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend/routes/store_api_data/')

def validate_api_key(api_key):
    """Loads existing API key data from api.json and validates the provided API key."""
    try:
        with open(f'{UPLOAD_FOLDER}api.json', 'r') as file:
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
        response = model.generate_content([image, full_prompt])
        
        return response.text
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while processing the image: {e}"
    
@request_model_bp.route('/request_iming', methods=['POST'])
def request_iming():
    api_key = request.form.get('api_key')
    prompt = request.form.get('prompt', "Say Hi!") 
    role = request.form.get('role', "'Mul-Model' AI Chatbot Bot")

    # Validate API key
    if not validate_api_key(api_key):
        return jsonify({"error": "Invalid API key"}), 400

    text_by_iming = process_iming(role, prompt)
    result = {"text": text_by_iming}
    return jsonify(result), 200


@request_model_bp.route('/request_analyze_image', methods=['POST'])
def request_analyze_image():
    api_key = request.form.get('api_key')
    prompt = request.form.get('prompt', "Process this image")
    role = request.form.get('role', "")

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