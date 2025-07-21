import hand_detector2 as hdm
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
from gtts import gTTS
import io
import pygame
import warnings
import joblib

# Ignore all warnings
warnings.filterwarnings("ignore")

"""Read and process data. The dataset 'hand_signals.csv' is loaded and any remaining unnamed columns are removed. 
The features (X) are extracted by dropping the target variable 'letter', and the target variable (y) is set to 'letter'."""

data = pd.read_csv('hand_signals.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

X = data.drop('letter', axis=1)
y = data['letter']

"""Split the data into training and test sets. This uses an 80-20 split where 80% of the data is used for training 
and 20% is reserved for testing."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Initialize and train the Logistic Regression model. The model is fitted with a iteration count of 200."""
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

"""The speech function converts the text input into speech using the Google Text-to-Speech (gTTS) library.
It loads the generated speech as an mp3 file and plays it using the pygame mixer."""
def speech(text):
    # Initializes the text
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    myobj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Load the BytesIO object as a sound
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()

    # Keep the program running while the sound plays
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

"""Main function initializes variables, hand detector, video capture, and timer. 
The while loop continuously processes frames from the camera, detects hand landmarks, and predicts the letter using the trained model. 
If inactivity is detected for more than 3 seconds, the word is spoken and added to a list."""
def main():
    pygame.mixer.init()
    signal_data = {}
    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()
    letters = [0]
    word = ''
    words = []
    start = time.time()
    end = time.time()

    """While loop to run the interpreter. Captures image, processes landmarks, and predicts letters from the hand detected. The word is built up over time, 
    and inactivity is detected with a timer."""
    while True:
        # Initialize image from the camera
        success, img = cap.read()
        img = cv2.flip(img, 1)
        key = cv2.waitKey(1) & 0xFF

        """The hand detector and position finder are initialized from the handDetector class. This extracts landmarks of the hand and uses them for predictions."""
        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)
        
        """A confidence threshold is set to 0.7 for the regression model to make sure predictions are reliable. If no landmarks are detected, the inactivity timer is triggered."""
        confidence_threshold = .7

        if not landmarks:
            start = time.time()
            idle_timer = start-end
            if idle_timer >= 3 and word != '':
                if word[-1] != ' ':
                    speech(word)
                    words.append(word)
                    word = word + ' '

        """If only one hand is detected, landmarks are used to extract the location vector. The model predicts the letter, and if the prediction probability exceeds 
        the threshold, the letter is displayed on the screen."""
        if landmarks and len(landmarks) == 1:
            lmlist = landmarks[0][1]
            end = time.time()

            p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25, min(lmlist[x][2] for x in range(len(lmlist))) - 25)
            p2 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25, max(lmlist[x][2] for x in range(len(lmlist))) + 25)
            cv2.rectangle(img, p1, p2, (255, 255, 255), 3)

            location_vector = np.array([coord for lm in lmlist for coord in lm[1:3]]).reshape(1, -1)
            
            probabilities = model.predict_proba(location_vector)
            max_prob = np.max(probabilities)
            if max_prob > confidence_threshold:
                predicted_letter = model.predict(location_vector)[0]
                if predicted_letter == letters[-1]:
                    letters.append(predicted_letter)
                else:
                    letters = [predicted_letter]
                cv2.putText(img, predicted_letter, (p1[0], p1[1] - 10), 
                            cv2.QT_FONT_NORMAL, 3, (255, 255, 255), 3)

            if len(letters) == 20:
                word = word + letters[0]
                letters = [0]
                print(word)

        # Show the image
        cv2.imshow("Image", img)

        """If 'c' is pressed, the current landmarks are saved into the signal_data dictionary for later use."""
        if key == ord('c') and lmlist:
            for item in lmlist:
                if f'{item[0]}x' in signal_data:
                    signal_data[f'{item[0]}x'].append(item[1])
                else:
                    signal_data[f'{item[0]}x'] = [item[1]]
                if f'{item[0]}y' in signal_data:
                    signal_data[f'{item[0]}y'].append(item[2])
                else:
                    signal_data[f'{item[0]}y'] = [item[2]]
        
        """If 'q' is pressed, the program is stopped and the video capture is released."""
        if key == ord('q'):
            break
        
    """If there is any signal data, it is added to the DataFrame and saved back to 'hand_signals.csv'."""
    if signal_data:
        signal_data['letter'] = ['a'] * len(signal_data['0x'])
        new_signals = pd.DataFrame(signal_data)
        existing_signals = pd.read_csv('hand_signals.csv')
        updated_stats = pd.concat([existing_signals, new_signals], ignore_index=True)
        updated_stats.to_csv('hand_signals.csv', index=False)

"""Runs the main function that processes the video and updates the words based on detected hand signals."""
if __name__ == '__main__':
    main()

joblib.dump(model, 'hand_sign_model.pkl')
