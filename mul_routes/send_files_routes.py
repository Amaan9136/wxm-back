from flask import Blueprint, send_from_directory, abort
import os

send_files_bp = Blueprint('send_files', __name__)

# To get a file from mul folder: /get_file/mul/yourfile.txt
# To get a file from api folder: /get_file/api/yourfile.txt
# To get a image from store_gen_img folder: /get_file/gen_img/{image_name_with_extension}

STORE_MUL_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_mulcmd_data/uploads/')
STORE_API_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_api_data/')
STORE_MODEL_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_api_data/models')
STORE_GEN_IMAGE = os.path.join(os.getcwd(), 'mul_routes/store_gen_img/')

@send_files_bp.route('/get_file/<folder_type>/<filename>', methods=['GET'])

def get_file(folder_type, filename):
       
    # Check if the requested filename is "api.json" in a case-insensitive manner
    if filename.lower() == "api.json":
        abort(403, description="Access to 'api.json' is forbidden.")
    if folder_type == 'mul':
        folder_path = STORE_MUL_FOLDER
    elif folder_type == 'api':
        folder_path = STORE_API_FOLDER
    elif folder_type == 'model':
        folder_path = STORE_MODEL_FOLDER
    elif folder_type == 'gen_img':
        folder_path = STORE_GEN_IMAGE
    else:
        abort(400, description="Invalid folder type specified")

    print(folder_path)

    file_path = os.path.join(folder_path, filename)
    
    print(file_path)

    if os.path.exists(file_path):
        return send_from_directory(directory=folder_path, path=filename)
    else:
        abort(404, description=f"File '{filename}' not found.")
