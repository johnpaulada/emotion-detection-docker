# Emotion Detection in Python
> Emotion detection from (almost) scratch in Python.

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)
[![forthebadge](http://forthebadge.com/images/badges/check-it-out.svg)](http://forthebadge.com)

## Install
### With Docker
1. Install Docker
2. Run `git clone https://github.com/johnpaulada/emotion-detection-docker.git` in your console
3. Run `cd emotion-detection-docker` in your console
4. Run `docker pull johnpaulada/emotion-detection` or `docker load -i emotion-detection.tar` in your console
5. Run `docker run -v 'pwd'/app:/home/app -it johnpaulada/emotion-detection` in your console
6. Run `cd app`
7. You're all set!

## Usage
For help, you can run `python3 app.py -h`.

### Extracting Data
To extract data, run `python3 app.py --data path-to-images-folder`.

For example:
```bash
python3 app.py --data ./images
```

### Training
To train the program, run `python3 app.py --train`.

### Prediction
To predict the emotions of input images, run `python3 app.py --predict path-to-input-image`.
The command supports the prediction of multiple images.

For example:
```bash
python3 app.py --predict ./happy-person.jpg ./angry_face.jpg
```

## Stack
1. Python
2. Numpy
3. OpenCV
4. dlib
5. imutils
6. Scikit-Learn
7. Docker

## License
MIT