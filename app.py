from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})

# WE XPERT MEET ROUTES
from wxm_routes.email_routes import email_routes
from wxm_routes.text_routes import text_routes
from wxm_routes.files_routes import files_routes
app.register_blueprint(email_routes)
app.register_blueprint(text_routes)
app.register_blueprint(files_routes)
# from routes.video_routes import video_routes
# from routes.face_routes import face_routes
# app.register_blueprint(video_routes)
# app.register_blueprint(face_routes)

# MUL MODEL ROUTES
from mul_routes.run_python_routes import run_python_bp
# from mul_routes.run_mulcmd_routes import run_mulcmd_bp
from mul_routes.send_files_routes import send_files_bp
from mul_routes.api_generate_routes import api_model_bp
from mul_routes.request_model_routes import request_model_bp
app.register_blueprint(run_python_bp)
# app.register_blueprint(run_mulcmd_bp)
app.register_blueprint(send_files_bp)
app.register_blueprint(api_model_bp)
app.register_blueprint(request_model_bp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/mul")
def mul():
    return render_template('mul.html')

@app.route("/wxm")
def wxm():
    return render_template('wxm.html')

if __name__ == '__main__':
    app.run(debug=True)