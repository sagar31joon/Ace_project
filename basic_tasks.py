#basic_tasks.py
#--------------
#Holds placeholder functions for ACE's basic tasks.
#These functions will be triggered by ACE based on the predicted intent.

import datetime

# =========================================================
# MAIN TASK ROUTER
# =========================================================

def run_task(intent, text):
    #Routes the detected intent to the correct task handler.
    #:param intent: The predicted intent label (string)
    #:param text: The user's original speech as text
    
    if intent == "WEB_SEARCH":
        web_search(text)
    
    elif intent == "OPEN_APP":
        open_app(text)
    
    elif intent == "Get_Time":
        get_time(text)

    elif intent == "MUTE":
        mute()
        
    elif intent == "UNMUTE":
        unmute() 
    
    else:
        unknown_intent(text)

# =========================================================
# PLACEHOLDER TASKS
# =========================================================

def web_search(query):
    print(f"[TASK] Would perform a web search for: {query}")

def open_app(app_name):
    print(f"[TASK] Would open application: {app_name}")

def get_time(query):                                                #Gets the current time and prints it.
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[TASK] Current Time is {current_time}")

def mute():
    print("[TASK] ACE is now muted.")

def unmute():
    print("[TASK] ACE is now unmuted.")

def unknown_intent(text):                                           #Handles cases where ACE doesn't recognize the intent.
    print(f"[TASK] Sorry, I didn't understand: {text}")




