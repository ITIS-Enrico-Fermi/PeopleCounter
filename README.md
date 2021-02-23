# Classifier
This repository is intended to be the second module of FaceCounter project (people counting and tracking)

## Setting up
### Windows
From CMD:
```
py -m venv venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
### Linux and MacOS
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt 
```

## Usage
### Save requirements
From Python Venv:
```
python -m pip freeze > requirements.txt
```
### How to run face detector?
```
python haar_cascade_classifier.py
```
Press _ESC_ to exit
