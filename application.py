from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, DataWrapper

application = Flask(__name__)

app = application

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # Render same webpage.
        return render_template('home.html')
    
    elif request.method == 'POST':
        
        data=DataWrapper(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
            )
        
        features = data.to_dataframe()

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(features)

        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0')
