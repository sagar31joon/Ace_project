#main_ace.py
#-----------
#Main controller for ACE.
#Connects input (voice/text), intent recognition, and task execution.

import joblib               #For loading/saving trained models
import os
import spacy                #creates sentence embeddings
import basic_tasks          #python file for tasks
import input_output as io   #python file for io control
import subprocess           #To re-run the training script if needed

# =========================================================
# 1. Load the intent classifier model
# =========================================================

print("[ACE] Loading Intent Classifier...")
intent_model = joblib.load("models/intent_classifier.pkl")

# =========================================================
# 2. Load SpaCy model for embeddings (must match training)
# =========================================================

print("[ACE] Loading Spacy Model...")
nlp = spacy.load("en_core_web_md")          ## Same model used when training

# =========================================================
# 3. Initialize speech recognition
# =========================================================

recognizer = io.init_speech_model(model_path="vosk_models/vosk-model-en-us-0.22")

print("[ACE] Ready. Press 'SHIFT + N' to start listening, 'SHIFT + M' to stop listening")

# =========================================================
# 4. Main Loop
# =========================================================

try:
    while True:
        # Get speech input only when listening is ON
        text = io.listen_and_transcribe(recognizer)
    
        if text:
            print(f"[User] {text}")
        
            # Convert user input to embeddings for the classifier
            doc = nlp(text)
            embeddings = doc.vector.reshape(1, -1)          # Reshape for scikit-learn
        
            # Predict intent
            predicted_intent = intent_model.predict(embeddings)[0]
            print(f"[ACE] Predicted Intent: {predicted_intent}")
        
            #Route to the correct task
            basic_tasks.run_task(predicted_intent, text)
except KeyboardInterrupt:
    print("\n [ACE] Shutting Down")