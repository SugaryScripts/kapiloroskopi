from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    image = FileField('Upload Image',
                     validators=[
                         FileRequired(message='Please select an image'),
                         FileAllowed(['jpg', 'jpeg', 'png'], message='Only JPG and PNG images are allowed!')
                     ])