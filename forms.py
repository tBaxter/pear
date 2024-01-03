from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired


class PearForm(FlaskForm):
    """
    Compare two blocks of text for textual relevancy.
    """
    text1 = TextAreaField('Put their text here', description='From a job description or any other text you want to analyze', validators=[DataRequired()])
    text2 = TextAreaField('Put your text here', description='From your resume or any other text you want to compare', validators=[DataRequired()])
    submit = SubmitField('Compare', description="Com-pear!")