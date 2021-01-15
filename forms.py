from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired


class PearForm(FlaskForm):
    """
    Compare two blocks of text for textual relevancy.
    """
    text1 = TextAreaField('Their text', description='(job description, whatever you want to analyze)', validators=[DataRequired()])
    text2 = TextAreaField('Your text', description='(your resume, whatever you want to compare)', validators=[DataRequired()])
    submit = SubmitField('Submit')