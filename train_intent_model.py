# ===============================
# Train the ACE Intent Classifier
# ===============================
# This script:
# 1. Reads our intent dataset from a CSV file.
# 2. Turns each sentence into a "meaning vector" using spaCy.
# 3. Trains a Logistic Regression model to classify intents.
# 4. Saves the trained model to disk for later use in ACE.
# ===============================

import pandas as pd             #for reading CSV file
import spacy                    #for creating sentence embeddings
import numpy as np              #for working with numericals arrays
from sklearn.linear_model import LogisticRegression #Classifier
from sklearn.model_selection import train_test_split #to split data into training and testing sets
from sklearn.metrics import classification_report #for evaluating the model perforance
import joblib                  #for saving and loading trained models
import os                      #for handeling file paths

# 1️⃣ Load spaCy's medium English model (includes good word vectors)
print("Loading spaCy model (This may take a few seconds)...")
nlp = spacy.load("en_core_web_md")

# 2️⃣ Load our dataset (CSV file containing text + intent labels)
dataset_path = "models/intent_dataset_10000_clean_2.csv"
print(f"Loading dataset from: {dataset_path}...")
df = pd.read_csv(dataset_path)

# 3️⃣ Turn each sentence into an embedding (numeric vector)
#    - spaCy's .vector gives a 300-dimensional representation of meaning
print("Creating sentence embeddings...")
x = np.array([nlp(text).vector for text in df["text"]])
y = df["intent"] # The labels (intents) for each sentence

# 4️⃣ Split the data into training set (90%) and testing set (10%)
#    - Training set is used to teach the model.
#    - Testing set is used to check how well it learned.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# 5️⃣ Create & train the Logistic Regression model
#    - max_iter=1000 allows enough passes over the data to converge.
print("Training Logistic Regression intent classifier...")
clf = LogisticRegression(
                            max_iter=1000,
                            C=0.5,          # lower = more regularization = less overfitting
                            penalty='l2',   # default, but we set explicitly
                            solver='lbfgs', # best for small/medium datasets, supports L2
                            multi_class='auto'
                        )
clf.fit(x_train, y_train)

# 6️⃣ Test the model on the testing set and print performance stats
print("\n Testing model on test data")
y_pred =clf.predict(x_test) #pred. short for predicted
print(classification_report(y_test, y_pred))

# 7️⃣ Save the trained model to a file so ACE can load it later
model_path = os.path.join("models", "intent_classifier.pkl")
joblib.dump(clf, model_path)
print(f"✅ Model saved to: {model_path}")

