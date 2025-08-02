# from flask import Flask, render_template, request
# import torch
# from transformers import BertTokenizer, BertModel
# from torch import nn
# import numpy as np

# print("🔥 Flask app is running...")


# app = Flask(__name__)

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# class BertClassifier(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(BertClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(128, 2)
#         self.relu = nn.ReLU()

#     def forward(self, input_id, mask):
#         _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.linear(dropout_output)
#         final_layer = self.relu(linear_output)
#         return final_layer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BertClassifier()
# model.load_state_dict(torch.load("bert_sarcasm_model.pt", map_location=device))
# model.to(device)
# model.eval()
#------------------------------------------------------------
# # Home route
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None

#     if request.method == "POST":
#         text = request.form["input_text"]

#         inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
#         input_id = inputs['input_ids'].to(device)
#         mask = inputs['attention_mask'].to(device)

#         with torch.no_grad():
#             output = model(input_id, mask)
#             probs = torch.softmax(output, dim=1).cpu().numpy()[0]
#             predicted_class = np.argmax(probs)
#             confidence = round(probs[predicted_class] * 100, 2)
#             prediction = "Sarcastic 🤨" if predicted_class == 1 else "Not Sarcastic 🙂"

#     return render_template("index.html", prediction=prediction, confidence=confidence)

# if __name__ == "__main__":
#     app.run(debug=True)


# First, let's save all your existing code up to the prediction function
# Then add the Flask app part below

# from flask import Flask, request, render_template
# import threading

# app = Flask(__name__)

# # Home route
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         text = request.form['headline']
#         prediction = predict_sarcasm(text)
#         return render_template('index.html', prediction=prediction, text=text)
#     return render_template('index.html')

# if __name__ == '__main__':
#     # Train model first (this will take some time)
#     print("Training model...")
#     # Make sure to run this only once! You might want to save your model after training
#     # and load it here instead of retraining every time
#     app.run(debug=True, use_reloader=False)

# app.py
from flask import Flask, request, render_template
from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])
app = Flask(__name__)

# --------------------- MODEL SETUP ---------------------
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(128, 2)
        
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        return self.linear(dropout_output)

# Initialize model
model = BertClassifier().to(device)
try:
    model.load_state_dict(torch.load("bert_sarcasm_model.pt", map_location=device))
    print("Loaded pretrained model successfully!")
except FileNotFoundError:
    print("No pretrained model found. Please train first!")
    exit()

model.eval()

# --------------------- HELPER FUNCTIONS ---------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z\s]+|X{2,}', '', text)
    text = re.sub("[@\^&\*\$]|#\S+|\S+[a-z0-9]\.(com|net|org)", " ", text)
    return text

# --------------------- FLASK ROUTES ---------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        text = request.form['headline']
        processed_text = preprocess_text(text)
        prediction = predict_sarcasm(processed_text)
    return render_template('index.html', prediction=prediction)

def predict_sarcasm(text: str) -> str:
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = model(inputs['input_ids'].to(device), 
                    inputs['attention_mask'].to(device))
    
    return "Sarcastic 🤨" if output[0].argmax().item() == 1 else "Not Sarcastic 🙂"

# --------------------- MAIN ---------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)