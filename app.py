from flask import Flask, request, jsonify
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizer

app = Flask(__name__)

# Configurar el modelo 
config = AutoConfig.from_pretrained('bert-base-uncased', num_labels=3)
model = AutoModelForSequenceClassification.from_config(config)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Iniciar el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Definimos ruta (0.0.0.0:5000/predict)
@app.route('/predict', methods=['POST'])
def predict():
    
    # Verificamos si tiene body
    data = request.json
    if 'mensaje' not in data:
        return jsonify({'error': 'No message provided'}), 400

    # Guardamos el mensaje
    mensaje = data['mensaje']
    
    # Tokenizamos el mensaje
    inputs = tokenizer(mensaje, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Las salidas son logits 
    logits = outputs.logits
    
    # Obtener la etiqueta predicha
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Mapear de vuelta a las etiquetas originales
    label_map = {
        0: "P+", 
        1: "P", 
        2: "NEU",
        3: "N",
        5: "N+",
        6: "NONE"
    }
    predicted_label = label_map[predicted_class]
    

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
