from flask import Flask, request, render_template
import pickle

# Load vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email']
        transformed_text = vectorizer.transform([email_text])
        pred = model.predict(transformed_text)[0]
        prediction = "Spam ❌" if pred == 1 else "Not Spam ✅"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
