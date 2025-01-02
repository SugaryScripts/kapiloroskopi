from flask import Flask
import onnxruntime as ort
import os

from kapiloroskopi.blueprints import main_menu


def start_app():
  app = Flask(__name__)

  register_config(app)
  register_blueprint(app)

  model_path = 'kapiloroskopi/storage/best.onnx'
  app.model = ort.InferenceSession(model_path)

  return app

def register_blueprint(app):
  app.register_blueprint(main_menu.blueprint)
# app.register_blueprint(dev_ground.blueprint)

def register_config(app):
  app.secret_key = os.getenv('SECRET_KEY', 'hello_there')
  flask_env = os.getenv('FLASK_ENV', 'development')
  app.debug = flask_env == 'development'
  app.config['UPLOAD_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')














