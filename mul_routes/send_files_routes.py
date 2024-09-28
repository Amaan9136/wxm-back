from flask import Blueprint, send_from_directory, abort
import os

send_files_bp = Blueprint('send_files', __name__)

STORE_MUL_FOLDER = os.path.join(os.getcwd(), 'mul_routes/store_mulcmd_data/uploads/')

@send_files_bp.route('/get_file/<filename>', methods=['GET'])
def get_file(filename):
    file_path = os.path.join(STORE_MUL_FOLDER, filename)
    
    def serve_file():
        if os.path.exists(file_path):
            return send_from_directory(directory=STORE_MUL_FOLDER, path=filename)
        else:
            abort(404, description=f"File '{filename}' not found.")
    
    return serve_file()
