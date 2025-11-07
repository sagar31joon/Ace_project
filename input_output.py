#input_output.py
#---------------
#Handles all input/output operations for ACE:
#- Microphone input (speech-to-text) using Vosk
#- Keyboard shortcuts to start/stop listening
#- Output to console (and later, TTS)

import queue                    # Thread-safe queue for passing audio data between mic callback and main loop
import json                     # To parse Vosk's speech recognition output (JSON format) into Python dicts
import sounddevice as sd        # For capturing real-time audio from the microphone
import vosk                     # Offline speech recognition library (loads model & converts audio → text)
from pynput import keyboard     # For global keyboard shortcut detection (Shift+N / Shift+M)
import time

# =========================================================
# Global variables
# =========================================================

listening = False       # Flag to track listening state
q = queue.Queue()       # Queue for audio data between callbacks & main loop

# =========================================================
# Audio callback (runs in the background when mic is active)
# =========================================================

def _audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio Status] {status}", flush=True)           #This function is called automatically whenever the microphone
    q.put(bytes(indata))                                        #receives new audio data. We put that audio into a queue so the
                                                                #main program can process it.
    
# =========================================================
# Keyboard handling
# =========================================================

def _on_press(key):                                  #Listens for keyboard key presses.
    global listening                                 #- Shift+N starts listening
    try:                                             #- Shift+M stops listening
        #Detect Shift+N
        if key == keyboard.KeyCode.from_char('N') and keyboard.Controller().pressed(keyboard.Key.shift):
            listening = True
            print("\n Listening is turned ON.. (SHIFT + N pressed)")
        #Detect Shift+M
        elif key == keyboard.KeyCode.from_char('M') and keyboard.Controller().pressed(keyboard.Key.shift):
            listening = False
            print("\n Listening is turned OFF.. (SHIFT + M pressed)")
    except Exception as e:
        print(f"[Keyboard Error] {e}")
        
# Start a background listener for Keyboard
_keyboard_listener = keyboard.Listener(on_press=_on_press)
_keyboard_listener.start()

# =========================================================
# Initialize Vosk speech recognizer
# =========================================================

def init_speech_model(model_path="vosk_models/vosk-model-en-us-0.22", samplerate=16000):
    
    print("[IO] Loading Vosk Speech Model...")                        #Loads the Vosk speech-to-text model.  
    model = vosk.Model(model_path)                                    #Returns the recognizer object.
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    return recognizer

# ==========================================
# Function: remove_leading_fillers
# Purpose: Removes filler words from the start of a transcribed sentence
# ==========================================
def remove_filling_fillers(text):
    fillers = {"the", "um", "uh", "like", "so", "hmm", "er", "ah"}
    words = text.split() # Split text into individual words
    while words and words[0].lower() in fillers: # Remove filler words from the beginning until a non-filler word is found
        words.pop(0)
    while words and words[-1].lower() in fillers:  # Remove fillers from end
        words.pop(-1)
    return " ".join(words)

# =========================================================
# Voice input function
# =========================================================
def listen_and_transcribe(recognizer, samplerate=16000, device=None, silence_timeout=1):
    
    #Continuously listens to the microphone when ACE is in listening mode,
    #processes audio using Vosk, and returns finalized recognized text.

    # Track the time we last heard any valid speech
    last_speech_time = time.time()

    # Open microphone stream for capturing audio in real-time
    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=2000,
        device=device,
        dtype='int16',
        channels=1,
        callback=_audio_callback
    ):
        while True:
            if listening:  # Only process audio when listening flag is True

                # =====================================================
                # 1️⃣ Silence flush check BEFORE touching the queue
                # If silent for more than `silence_timeout` seconds,
                # clear old audio chunks so we don’t process stale silence.
                # =====================================================
                if time.time() - last_speech_time > silence_timeout:
                    if not q.empty():  # Only clear if there's data
                        q.queue.clear()
                        print("[IO] Cleared old silence from queue")
                    last_speech_time = time.time()  # Reset timer

                try:
                    # Get audio data from Queue (waits up to 0.1 sec)
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    continue  # No data yet, keep looping

                # =====================================================
                # 2️⃣ Pass the audio chunk to the Vosk Recognizer
                # Only finalized speech results are considered
                # =====================================================
                if recognizer.AcceptWaveform(data):
                    last_speech_time = time.time()  # Reset when full speech detected

                    # Convert recognizer result from JSON to Python dictionary
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    # Remove any leading/trailing filler words
                    text = remove_filling_fillers(text)

                    if text:
                        return text  # ✅ Return recognized speech as text

            else:
                # When not listening, sleep briefly to avoid wasting CPU
                time.sleep(0.1)
                
# =========================================================
# Output function (for now, just console print)
# =========================================================

def output_text(message):
    print (f"ACE : {message}")                          #Sends output from ACE to the user.
                                                        #Currently: console print
                                                        #Later: could be expanded to text-to-speech.