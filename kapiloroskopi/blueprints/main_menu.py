from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, current_app, flash, session, redirect, url_for, send_from_directory, \
  jsonify, request
from kapiloroskopi.app.forms import PredictionForm
import pandas as pd
import numpy as np
import os
import torch

blueprint = Blueprint("menu", __name__)


@blueprint.route("/", methods=['GET', 'POST'])
def index():
  form = PredictionForm()
  prediction_result = None
  confidence_normal = None
  confidence_abnormal = None
  confidence = None
  form_errors = None

  # Calculate or retrieve model performance metrics
  model_precision_0 = 0.83  # Example value
  model_precision_1 = 0.76  # Example value
  model_recall_0 = 0.73  # Example value
  model_recall_1 = 0.85  # Example value
  model_f1_score_0 = 0.78  # Example value
  model_f1_score_1 = 0.81  # Example value
  model_accuracy = 0.79  # Example value

  if form.is_submitted():
      if form.validate_on_submit():
          # Get the uploaded image
          image_file = form.image.data
          filename = secure_filename(image_file.filename)
          filepath = os.path.join(current_app.config['UPLOAD_PATH'], filename)

          try:
              # Save the image temporarily
              image_file.save(filepath)

              # Preprocess image using your function
              img_tensor = preprocess_image(filepath)

              # Run prediction using ONNX session
              outputs = current_app.model.run(None, {current_app.model.get_inputs()[0].name: img_tensor})
              output_tensor = torch.tensor(outputs[0])
              prediction = output_tensor.argmax().item()

              # Get probabilities
              probabilities = torch.softmax(output_tensor, dim=1)[0]
              confidence = float(probabilities[prediction])
              confidence = round(confidence * 100, 2)

              # Get individual class probabilities
              confidence_normal = round(float(probabilities[0]) * 100, 2)
              confidence_abnormal = round(float(probabilities[1]) * 100, 2)

              print(f"Predicted {prediction} - {confidence}")

              # Convert prediction to class name
              prediction_result = "Non Neuropathy" if prediction == 0 else "Neuropathy"

              return render_template('menu/dashboard.html',
                                     form=form,
                                     prediction_result=prediction_result,
                                     confidence=confidence,
                                     confidence_normal=confidence_normal,
                                     confidence_abnormal=confidence_abnormal)

          finally:
              # Clean up - remove temporary file
              if os.path.exists(filepath):
                  os.remove(filepath)

      else:
          if not form.validate():
              form_errors = [f"{field.label.text}: {' '.join(field.errors)}" for field in form if field.errors]

      return render_template('menu/dashboard.html',
                             form=form,
                             form_errors=form_errors if 'form_errors' in locals() else None)

  return render_template('menu/dashboard.html',
                         page_name='index', form=form,
                         prediction_result=prediction_result,
                         form_errors = form_errors,
                         confidence_normal=confidence_normal,
                         confidence_abnormal=confidence_abnormal,
                         confidence = confidence,
                         model_precision_0=model_precision_0,
                         model_precision_1=model_precision_1,
                         model_recall_0=model_recall_0,
                         model_recall_1=model_recall_1,
                         model_f1_score_0=model_f1_score_0,
                         model_f1_score_1=model_f1_score_1,
                         model_accuracy=model_accuracy
                         )


import cv2

def preprocess_image(img_path, input_shape=(640, 640)):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = img.transpose(2, 0, 1).astype('float32')
    img /= 255.0
    return np.expand_dims(img, axis=0)