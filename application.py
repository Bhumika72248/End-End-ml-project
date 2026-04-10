import logging
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        logger.info("Input data received: %s", pred_df.to_dict())

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logger.info("Prediction successful: %s", results[0])

        return render_template('home.html', results=results[0])

    except ValueError as e:
        logger.error("Invalid input data: %s", str(e))
        return render_template('home.html', error="Invalid input: please check all fields are filled correctly.")

    except Exception as e:
        logger.exception("Prediction failed")
        return render_template('home.html', error="Prediction failed due to a server error. Please try again later.")


@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal server error: %s", str(error))
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0")
