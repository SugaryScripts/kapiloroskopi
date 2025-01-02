from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, current_app, flash, session, redirect, url_for, send_from_directory, \
    jsonify, request
from kapiloroskopi.app.forms import PredictionForm
import pandas as pd
import numpy as np
import os
import torch
import base64

blueprint = Blueprint("menu", __name__)


@blueprint.route("/", methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    prediction_result = None
    confidence_neuro = None
    confidence_non = None
    confidence = None
    form_errors = None
    preview_image = None

    # Calculate or retrieve model performance metrics
    model_precision_0 = 0.88  # Example value
    model_precision_1 = 0.85  # Example value
    model_precision_2 = 0.79  # Example value
    model_recall_0 = 0.91  # Example value
    model_recall_1 = 0.85  # Example value
    model_recall_2 = 0.84  # Example value
    model_f1_score_0 = 0.89  # Example value
    model_f1_score_1 = 0.85  # Example value
    model_f1_score_2 = 0.81  # Example value
    model_accuracy_0 = 0.89  # Example value
    model_accuracy_1 = 0.85  # Example value
    model_accuracy_2 = 0.82  # Example value
    model_specificity_0 = 0.88  # Example value
    model_specificity_1 = 0.85  # Example value
    model_specificity_2 = 0.81  # Example value

    if form.is_submitted():
        if form.validate_on_submit():
            # Get the uploaded image
            image_file = form.image.data
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_PATH'], filename)

            try:
                # Save the image temporarily
                image_file.save(filepath)

                # Convert image to base64 for preview
                with open(filepath, 'rb') as img_file:
                    preview_image = base64.b64encode(img_file.read()).decode('utf-8')
                    preview_image = f"data:image/jpeg;base64,{preview_image}"

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
                confidence_neuro = round(float(probabilities[0]) * 100, 2)
                confidence_non = round(float(probabilities[1]) * 100, 2)

                print(f"Predicted {prediction} - {confidence}")

                # Convert prediction to class name
                prediction_result = "Neuropathy" if prediction == 0 else "Non Neuropathy"

                return render_template('menu/dashboard.html',
                                       page_name='index',
                                       form=form,
                                       prediction_result=prediction_result,
                                       confidence=confidence,
                                       confidence_neuro=confidence_neuro,
                                       confidence_non=confidence_non,
                                       preview_image=preview_image,
                                       model_precision_0=model_precision_0,
                                       model_precision_1=model_precision_1,
                                       model_precision_2=model_precision_2,
                                       model_recall_0=model_recall_0,
                                       model_recall_1=model_recall_1,
                                       model_recall_2=model_recall_2,
                                       model_f1_score_0=model_f1_score_0,
                                       model_f1_score_1=model_f1_score_1,
                                       model_f1_score_2=model_f1_score_2,
                                       model_accuracy_0=model_accuracy_0,
                                       model_accuracy_1=model_accuracy_1,
                                       model_accuracy_2=model_accuracy_2,
                                       model_specificity_0=model_specificity_0,
                                       model_specificity_1=model_specificity_1,
                                       model_specificity_2=model_specificity_2
                                       )

            finally:
                # Clean up - remove temporary file
                if os.path.exists(filepath):
                    os.remove(filepath)

        else:
            if not form.validate():
                form_errors = [f"{field.label.text}: {' '.join(field.errors)}" for field in form if field.errors]

        return render_template('menu/dashboard.html',
                               page_name='index',
                               form=form,
                               form_errors=form_errors if 'form_errors' in locals() else None,
                               model_precision_0=model_precision_0,
                               model_precision_1=model_precision_1,
                               model_precision_2=model_precision_2,
                               model_recall_0=model_recall_0,
                               model_recall_1=model_recall_1,
                               model_recall_2=model_recall_2,
                               model_f1_score_0=model_f1_score_0,
                               model_f1_score_1=model_f1_score_1,
                               model_f1_score_2=model_f1_score_2,
                               model_accuracy_0=model_accuracy_0,
                               model_accuracy_1=model_accuracy_1,
                               model_accuracy_2=model_accuracy_2,
                               model_specificity_0=model_specificity_0,
                               model_specificity_1=model_specificity_1,
                               model_specificity_2=model_specificity_2
                               )

    return render_template('menu/dashboard.html',
                           page_name='index',
                           form=form,
                           prediction_result=prediction_result,
                           form_errors=form_errors,
                           confidence_neuro=confidence_neuro,
                           confidence_non=confidence_non,
                           confidence=confidence,
                           preview_image=preview_image,
                           model_precision_0=model_precision_0,
                           model_precision_1=model_precision_1,
                           model_precision_2=model_precision_2,
                           model_recall_0=model_recall_0,
                           model_recall_1=model_recall_1,
                           model_recall_2=model_recall_2,
                           model_f1_score_0=model_f1_score_0,
                           model_f1_score_1=model_f1_score_1,
                           model_f1_score_2=model_f1_score_2,
                           model_accuracy_0=model_accuracy_0,
                           model_accuracy_1=model_accuracy_1,
                           model_accuracy_2=model_accuracy_2,
                           model_specificity_0=model_specificity_0,
                           model_specificity_1=model_specificity_1,
                           model_specificity_2=model_specificity_2
                           )


import cv2


def preprocess_image(img_path, input_shape=(640, 640)):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = img.transpose(2, 0, 1).astype('float32')
    img /= 255.0
    return np.expand_dims(img, axis=0)
