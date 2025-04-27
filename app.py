from flask import Flask, request, jsonify, render_template
import joblib
import os
app = Flask(__name__)

loaded_vectorizer = joblib.load('vectorizer_h_final.pkl')
loaded_model = joblib.load('stacking_model_h_final.pkl')

def classify_news(news_text):
    text_vector = loaded_vectorizer.transform([news_text])
    predicted_category = loaded_model.predict(text_vector)
    return predicted_category[0]

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('news_text')

    if not news_text:
        return jsonify({'error': 'No news_text provided'}), 400

    predicted_category = classify_news(news_text)
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)