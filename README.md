## ONNX MODEL FLASK + Docker App

**This app is using Flask 1.1.2 and Python 3.8.8**. 

## Files
#### model.py
**contains the python onnx model for transforming image**
- python3 model.py to run

#### app.py
**contains the flask app of the model**
- python3 app.py to run 

#### templates 
**contains html templats for running the flask app**
- index.html

#### Dockerfile
**built docker image from ubuntu18.04. To build cocker image and run it use the commands below**
- _docker build ._
- _docker run -p_ 

#### requirements.txt
**dependencies for running the app**
- to generate the requirements file use the command below
- pipreqs  /"Working Directory"/

## Tech stack
Ubuntu 18.04
Python
Flask 
Docker

**All computations were done on ubuntu 18.04**

## Running this app

You'll need to have [Docker installed](https://docs.docker.com/get-docker/).
It's available on Windows, macOS and most distros of Linux. 

#### Check it out in a browser:

Visit <http://localhost:8000> in your favorite browser.







