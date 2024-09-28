import json, os
import secrets
import string
from flask import Blueprint, jsonify, request

api_model_bp = Blueprint('api_model_bp', __name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_api_data/')

def generate_api_key(name, length=24):
    # Take the first three letters from the name
    prefix = name[:3] if name else "XYZ"  # Default to "XYZ" if name is too short
    remaining_length = length - len(prefix)  # Calculate remaining length for random part

    # Generate the random part
    characters = string.ascii_letters + string.digits  # Letters and digits
    random_part_length = remaining_length // 2
    suffix_length = remaining_length - random_part_length  # Remainder for the suffix

    random_part = ''.join(secrets.choice(characters) for _ in range(random_part_length))
    suffix_part = ''.join(secrets.choice(characters) for _ in range(suffix_length))

    # Construct the full API key
    return random_part + prefix + suffix_part

def load_existing_data():
    try:
        with open(f'{UPLOAD_FOLDER}api.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file does not exist

def save_to_api_json(name, email, api_key):
    existing_data = load_existing_data()

    # Append new entry
    existing_data.append({"name": name, "email": email, "generated_api_key": api_key})

    # Save back to the file
    with open(f'{UPLOAD_FOLDER}api.json', 'w') as file:
        json.dump(existing_data, file, indent=4)

def generate_and_save_api_key(name, email):
    existing_data = load_existing_data()

    # Check if the email already exists
    for entry in existing_data:
        if entry['email'] == email:
            return {
                "message": "API KEY already exists",
                "name": entry['name'],  
                "email": email,
                "generated_api_key": entry['generated_api_key']
            }

    # Generate a unique API key
    generated_api_key = generate_api_key(name)

    # Save the new entry
    save_to_api_json(name, email, generated_api_key)

    return {
        "message": "API KEY Generated",
        "name": name,
        "email": email,
        "generated_api_key": generated_api_key
    }

@api_model_bp.route('/generate_api', methods=['GET', 'POST'])
def get_api_info():
    if request.method == 'GET':
        email = request.args.get('email')
        name = request.args.get('name')
        if email:
            email = email.strip('"')
        result = generate_and_save_api_key(name, email)
        return jsonify(result), 200

    elif request.method == 'POST':
        json_data = request.get_json()
        email = json_data.get('email') if json_data else None
        name = json_data.get('name') if json_data else None
        result = generate_and_save_api_key(name, email)
        return jsonify(result), 200
