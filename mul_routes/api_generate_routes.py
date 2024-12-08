import json
import os
import re
import secrets
import string
from flask import Blueprint, jsonify, request

api_model_bp = Blueprint('api_model_bp', __name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend/routes/store_api_data/')

def load_existing_data(path):
    try:
        with open(os.path.join(UPLOAD_FOLDER, path), 'r') as file:
            content = file.read().strip()
            if not content:  # Check if the content is empty
                return []
            return json.loads(content)  # Use loads instead of load
    except FileNotFoundError:
        return [] 
    except json.JSONDecodeError:  # Catch JSON decoding errors
        return [] 

# Save data to a specified file (api.json or deployed_models.json)
def save_data(path, data):
    with open(os.path.join(UPLOAD_FOLDER, path), 'w') as file:
        json.dump(data, file, indent=4)

def generate_api_key(name, length=24):
    prefix = name[:3].lower() if name else "xyz"
    remaining_length = length - len(prefix)

    characters = string.ascii_letters + string.digits
    random_part_length = remaining_length // 2
    suffix_length = remaining_length - random_part_length

    random_part = ''.join(secrets.choice(characters) for _ in range(random_part_length))
    suffix_part = ''.join(secrets.choice(characters) for _ in range(suffix_length))

    return random_part + prefix + suffix_part

def generate_and_save_api_key(name, email):
    existing_data = load_existing_data('api.json')

    # Check if the email already exists
    for entry in existing_data:
        if entry['email'].lower() == email.lower():
            return {
                "message": "API KEY already exists!",
                "name": entry['name'],
                "email": email,
                "generated_api_key": entry['generated_api_key']
            }

    generated_api_key = generate_api_key(name)

    # Save the new entry
    existing_data.append({"name": name, "email": email, "generated_api_key": generated_api_key})
    save_data('api.json', existing_data)

    return {
        "message": "API KEY Generated!",
        "name": name,
        "email": email,
        "generated_api_key": generated_api_key
    }

@api_model_bp.route('/generate_api', methods=['POST'])
def get_api_info():
    json_data = request.get_json()
    email = json_data.get('email') if json_data else None
    name = json_data.get('name') if json_data else None

    if not name or not email:
        return jsonify({"message": "Name and Email are required."}), 400

    result = generate_and_save_api_key(name, email)
    return jsonify(result), 200

def save_model_file(file, filename):
    model_path = os.path.join(UPLOAD_FOLDER, "models/"+filename)
    file.save(model_path)

@api_model_bp.route('/deploy_model', methods=['POST'])
def deploy_model():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    title = request.form.get('title', '')
    api_key = request.form.get('api_key', '')
    description = request.form.get('description', '')

    if not title or not api_key or not description:
        return jsonify({"message": "All fields are required."}), 400

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and file.filename.endswith('.pkl'):
        api_data = load_existing_data('api.json')
        matched_entry = next((entry for entry in api_data if entry['generated_api_key'] == api_key), None)

        if not matched_entry:
            return jsonify({"message": "Invalid API key"}), 400

        model_name = re.sub(r'\s+', '-', title.strip().lower())

        deployed_data = load_existing_data('deployed_models.json')
        if any(entry['path'] == model_name for entry in deployed_data):
            return jsonify({"message": "Model already exists! Please provide another title."}), 400

        save_model_file(file, model_name + ".pkl")

        deployed_data.append({
            "name": matched_entry['name'],
            "email": matched_entry['email'],
            "title": title.strip(),
            "api_key": api_key,
            "description": description,
            "model_path": model_name + ".pkl",
            "path": model_name
        })
        save_data('deployed_models.json', deployed_data)

        return jsonify({
            "message": "Model deployed successfully!",
            "name": matched_entry['name'],
            "path": model_name
        }), 200
    else:
        return jsonify({"message": "Invalid file format. Please upload a .pkl file."}), 400

@api_model_bp.route('/remove_deployed_model', methods=['DELETE'])
def remove_deployed_model():
    json_data = request.get_json()
    model_path = json_data.get('path') if json_data else None

    if not model_path:
        return jsonify({"message": "Model path is required."}), 400

    # Load the existing deployed models data
    deployed_data = load_existing_data('deployed_models.json')

    # Check if the model exists based on the provided path
    model_entry = next((entry for entry in deployed_data if entry['path'] == model_path), None)

    if not model_entry:
        return jsonify({"message": "Model not found."}), 404

    # Remove the model entry from the deployed data
    deployed_data.remove(model_entry)
    save_data('deployed_models.json', deployed_data)

    # Delete the model file from the filesystem
    model_file_path = os.path.join(UPLOAD_FOLDER, "models", model_path + ".pkl")
    if os.path.exists(model_file_path):
        os.remove(model_file_path)

    return jsonify({"message": f"Model {model_path} removed successfully!"}), 200
