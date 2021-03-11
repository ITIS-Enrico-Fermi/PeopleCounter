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
python haar_cascade_classifier.py --source dataset/maia.mp4 --processed-frame-preview
```
Press _ESC_ to exit

### Args
| short arg |            arg            |                action               |
|:---------:|:-------------------------:|:-----------------------------------:|
| -h        | --help                    | Show help message and exit          |
|           | --model MODEL             | Cascade classifier model name       |
|           | --source SOURCE           | Camera number or video filename     |
|           | --image IMAGE             | Image filename                      |
|           | --processed-frame-preview | Show the preview of processed frame |

## Performance
Original video's size is Full HD
|               Description              | MIN [s] | MAX [s] | AVG [s] |
|:--------------------------------------:|:-------:|:-------:|:-------:|
| Grayscale + hist equalization          |   0.80  |   1.90  |   1.61  |
| previous + downscale to VGA resolution |   0.22  |   0.22  |   0.11  |
| scaleFactor = 1.2                      |   0.13  |   0.13  |   0.07  |
