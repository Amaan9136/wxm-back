# backend/routes/run_mulcmd.py

from flask import Blueprint, request, jsonify
from .store_mulcmd_data.mul_cmd import MulCmd
from .store_mulcmd_data.helpers import parse_value
from .run_python_routes import execute_code
import os, json
from werkzeug.utils import secure_filename

run_mulcmd_bp = Blueprint('run_mulcmd', __name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend/routes/store_mulcmd_data/uploads/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_mulcmd_commands(commands, mulcmd_instance):
    output_messages = []
    processed_file_paths = {
        "plot_data_path":"",
        "plot_predict_path":"",
        "model_path":""
    }
    filenames = []

    for command in commands.split('\n'):
        command = command.strip()
        if command:
            parts = command.split()
            cmd = parts[0].lower()
            try:
                if cmd == 'file':
                    # Expecting: file "filename1.csv","filename2.csv"
                    filenames = ' '.join(parts[1:]).replace('"', '').split(',')
                    output_messages.append(mulcmd_instance.file(filenames))
                elif cmd == 'encode_features':
                    # Expecting: encode_features label=1 2 onehot=3 4
                    encoding_command = ' '.join(parts[1:])
                                        
                    # Validate and parse the encoding command
                    if encoding_command:
                        output_messages.append(mulcmd_instance.encode_features(encoding_command))
                    else:
                        output_messages.append("Error: 'encode_features' command requires encoding type and columns.")

                elif cmd == 'features':
                    # Expecting: features start [end]
                    if len(parts) < 2:
                        output_messages.append("Error: 'features' command requires at least one argument.")
                        continue

                    try:
                        start = int(parts[1])
                        end = int(parts[2]) if len(parts) > 2 else None
                        
                        # Call the features_range method with the parsed values
                        output_messages.append(mulcmd_instance.features_range(start, end))

                    except ValueError:
                        output_messages.append("Error: Invalid start or end value. Please ensure they are integers.")

                elif cmd == 'target':
                    # Expecting: target index1 [index2 ...]
                    if len(parts) < 2:
                        output_messages.append("Error: 'target' command requires at least one argument.")
                        continue
                    columns = parts[1:]
                    columns = [int(col) for col in columns]
                    output_messages.append(mulcmd_instance.target_range(*columns))
                elif cmd == 'split':
                    # Expecting: split ratio
                    if len(parts) != 2:
                        output_messages.append("Error: 'split' command requires exactly one argument.")
                        continue
                    ratio = float(parts[1])
                    output_messages.append(mulcmd_instance.split(ratio))
                
                elif cmd == 'model':
                    # Expecting: model "model_name" [arg1=value1 arg2=value2 ...]
                    if len(parts) < 2:
                        output_messages.append("Error: 'model' command requires at least one argument (model name).")
                        continue
                    
                    model_name = parts[1].replace('"', '')
                    # Collect remaining arguments as key-value pairs
                    arguments = {}
                    if len(parts) > 2:
                        for arg in parts[2:]:
                            try:
                                key, value = arg.split('=')
                                # Parse value to handle numerical data
                                arguments[key] = parse_value(value)
                            except ValueError:
                                output_messages.append(f"Error: Argument '{arg}' is not in the format key=value.")
                                continue
                    
                    output_messages.append(mulcmd_instance.set_model(model_name, **arguments))

                elif cmd == 'print_predict':
                    output_messages.append(mulcmd_instance.print_predict())
                elif cmd == 'print_acc':
                    output_messages.append(mulcmd_instance.print_accuracy())
                elif cmd == 'print_mse':
                    output_messages.append(mulcmd_instance.print_mse())
                elif cmd == 'print_mae':
                    output_messages.append(mulcmd_instance.print_mae())
                elif cmd == 'print_cm':
                    output_messages.append(mulcmd_instance.print_confusion_matrix())
                elif cmd == 'print_r2':
                    output_messages.append(mulcmd_instance.print_r2())
                elif cmd == 'plot_data':
                    plot_data_path,output_message=mulcmd_instance.plot_data()
                    output_messages.append(output_message)
                    processed_file_paths["plot_data_path"] = plot_data_path
                elif cmd == 'plot_predict':
                    plot_predict_path,output_message=mulcmd_instance.plot_predict()
                    output_messages.append(output_message)
                    processed_file_paths["plot_predict_path"] = plot_predict_path
                elif cmd == 'save_model':
                    # Expecting: save_model "filename.pkl"
                    if len(parts) != 2:
                        output_messages.append("Error: 'save_model' command requires exactly one argument.")
                        continue
                    filename = parts[1].replace('"', '')
                    model_path,output_message=mulcmd_instance.save_model(filename)
                    output_messages.append(output_message)
                    processed_file_paths["model_path"] = model_path
                elif cmd == 'model_predict':
                    # Expecting: model_predict "model_name" [[], []]
                    if len(parts) < 3:
                        output_messages.append('Error: "model_predict" command requires at least two arguments (Expecting: model_predict "SAVED_model_name" [[], []]).')
                        continue
                    model_name = parts[1].replace('"', '')
                    prediction_data_str = ' '.join(parts[2:])
                    prediction_data = json.loads(prediction_data_str)
                    # Ensure prediction data is a valid 2D array
                    if not isinstance(prediction_data, list) or not all(isinstance(item, list) for item in prediction_data):
                        output_messages.append("Error: 'model_predict' command expects a 2D array as prediction data.")
                        continue
                    predictions = mulcmd_instance.model_predict(model_name, prediction_data)
                    output_messages.append(f"Predictions for model '{model_name}':\nData: {prediction_data}\nPredicted: {predictions}")

                else:
                    # Execute custom Python command
                    result = execute_code(command)
                    output_messages.append(f"Executed custom Python command: {command}. Result: {result}")
            except Exception as e:
                output_messages.append(f"Error processing command '{command}': {str(e)}")
    return filenames, output_messages, processed_file_paths

@run_mulcmd_bp.route('/run_mulcmd', methods=['POST'])
def run_mulcmd_code():
    code = request.form.get('code', '')
    files = request.files.getlist('files')

    mulcmd_instance = MulCmd()
    output_messages = []
    allowed_filenames = []

    # First, save the uploaded files
    saved_files = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                file.save(file_path)
                saved_files.append(filename)
                output_messages.append(f"File '{filename}' saved successfully.")
            except Exception as e:
                output_messages.append(f"Failed to save file '{filename}'. Error: {str(e)}")
        elif file.filename:
            output_messages.append(f"File '{file.filename}' ignored as it's not allowed.")

    # Then, process the commands
    if code:
        try:
            allowed_filenames, command_outputs, processed_file_paths = process_mulcmd_commands(code, mulcmd_instance)
            # Ensure command_outputs is a list of strings
            if not isinstance(command_outputs, list):
                command_outputs = [str(command_outputs) if command_outputs is not None else '']
            else:
                command_outputs = [str(item) if item is not None else '' for item in command_outputs]

            output_messages.extend(command_outputs)
        except Exception as e:
            output_messages.append(f"Error processing commands: {str(e)}")

    # Check if any files mentioned in commands were not uploaded
    for filename in allowed_filenames:
        if filename not in saved_files:
            output_messages.append(f"File '{filename}' mentioned in commands was not uploaded.")

    # Join messages into a single string
    output = "\n".join(output_messages)
    return jsonify({'output': output, 'file_paths': processed_file_paths})

@run_mulcmd_bp.route('/run_all_mulcmd', methods=['POST'])
def run_all_mulcmd_code():
    cells = request.form.get('cells', '[]')
    cells = json.loads(cells)
    files = request.files.getlist('files')

    mulcmd_instance = MulCmd()
    outputs = []
    file_paths = []

    # First, save all uploaded files
    saved_files = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                file.save(file_path)
                saved_files.append(filename)
            except Exception as e:
                outputs.append(f"Failed to save file '{file.filename}'. Error: {str(e)}")
        elif file.filename:
            outputs.append(f"File '{file.filename}' ignored as it's not allowed.")

    # Process each code cell and collect output and file paths
    for index, code in enumerate(cells):
        if not isinstance(code, str):
            outputs.append(f"Code Cell {index + 1}: Invalid code cell format. Each cell should be a string.")
            file_paths.append({})
            continue
        
        try:
            allowed_filenames, command_outputs, processed_file_paths = process_mulcmd_commands(code, mulcmd_instance)
            # Ensure command_outputs is a list of strings, converting None to empty strings
            if not isinstance(command_outputs, list):
                command_outputs = [str(command_outputs) if command_outputs is not None else '']
            else:
                command_outputs = [str(item) if item is not None else '' for item in command_outputs]

            # Add output with cell index
            outputs.append(f"Code Cell {index + 1} Output:\n" + "\n".join(command_outputs))
            file_paths.append(processed_file_paths)  # Append file paths for this cell
        except Exception as e:
            outputs.append(f"Code Cell {index + 1}: Error processing commands. Details: {str(e)}")
            file_paths.append({})

    return jsonify({'outputs': outputs, 'file_paths': file_paths})
