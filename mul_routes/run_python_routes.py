from flask import Blueprint, request, jsonify
import subprocess
import json
import os
import re
import tempfile

# Define constants for file paths
VARIABLES_FILE = os.path.join(tempfile.gettempdir(), 'variables.json')
FUNCTIONS_FILE = os.path.join(tempfile.gettempdir(), 'functions.py')

run_python_bp = Blueprint('run_python', __name__)

# Load existing variables from JSON file
def load_variables():
    if os.path.exists(VARIABLES_FILE):
        try:
            with open(VARIABLES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Save variables to JSON file
def save_variables(variables):
    with open(VARIABLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(variables, f, indent=4)

# Load stored functions from a Python file
def load_functions():
    if os.path.exists(FUNCTIONS_FILE):
        with open(FUNCTIONS_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Save functions to a Python file
def save_function(function_code):
    with open(FUNCTIONS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{function_code}\n")

# Execute code and manage variable and function persistence
def execute_code(code):
    variables = load_variables()
    functions = load_functions()

    # Generate the code to load variables and functions into the execution environment
    load_vars_code = "\n".join([f"{key} = {json.dumps(value)}" for key, value in variables.items()])
    
    # Append the user code to the loaded variables and functions
    full_code = f"{load_vars_code}\n{functions}\n{code}"

    try:
        # Execute the code using the default Python interpreter in Vercel
        process = subprocess.Popen(['python', '-c', full_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()

        if error:
            return error

        # Extract functions from the code
        new_functions = extract_functions(code)
        existing_functions = load_functions()

        # Save new functions or replace existing ones
        for func_code in new_functions:
            func_name = extract_function_name(func_code)
            if f"def {func_name}(" not in existing_functions:
                save_function(func_code)

        # Save updated variables to the JSON file
        exec_globals = {}
        exec(full_code, globals(), exec_globals)
        new_variables = {key: exec_globals[key] for key in exec_globals if not key.startswith('__') and not callable(exec_globals[key])}
        save_variables(new_variables)

        # Check if the output is empty
        if not output.strip():
            output = "Cell executed without errors"

        return output
    except Exception as e:
        return str(e)

def extract_functions(code):
    """
    Extract function definitions from code.
    """
    func_pattern = re.compile(r'def .+?:\n(?:    .*\n)*')
    return func_pattern.findall(code)

def extract_function_name(func_code):
    """
    Extract function name from the function code.
    """
    match = re.match(r'def (\w+)\(', func_code)
    return match.group(1) if match else ""

# Blueprint routes
@run_python_bp.route('/run_python', methods=['POST'])
def run_code():
    code = request.form.get('code', '')

    output = execute_code(code)

    return jsonify({'output': output})

@run_python_bp.route('/run_all_python', methods=['POST'])
def run_all_code():
    cells = request.form.get('cells', '[]')
    cells = json.loads(cells)

    outputs = []
    for code in cells:
        output = execute_code(code)
        outputs.append(output)

    # Replace empty outputs with a default message
    outputs = [output if output.strip() else "Cell executed without errors" for output in outputs]

    return jsonify({'outputs': outputs})
