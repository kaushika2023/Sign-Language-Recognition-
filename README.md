# Sign-Language-Recognition-
Sign Language Recognition System A deep learning-based project that translates sign language gestures into text/speech to bridge communication gaps between hearing-impaired individuals and others. 

Features

✨ Real-time Recognition: Captures and interprets sign language gestures using webcam
📝 Text Translation: Converts signs to readable text
🗣️ Speech Output: Provides audible translation of signs
📊 Multi-Sign Support: Recognizes alphabets, numbers, and common phrases
🌐 Web Interface: User-friendly Flask-based web application

Impact: Promotes inclusive communication with 92% accuracy on common signs.

Dataset

The model was trained on:

Custom collected dataset of 5,000+ sign samples

ASL (American Sign Language) benchmark dataset

ISL (Indian Sign Language) samples
Model Architecture
The system uses a hybrid deep learning approach:

CNN for spatial feature extraction from frames

LSTM for temporal sequence analysis

Combined using attention mechanism.


Future Work

Expand vocabulary to 500+ signs

Add sentence-level recognition

Develop mobile application

Implement two-way translation (voice to sign)
