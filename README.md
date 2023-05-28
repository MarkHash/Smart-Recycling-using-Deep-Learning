# Smart-Recycling-using-Deep-Learning
Smart Recycling using Deep Learning

This project is to develop image classification problem that will predict whether the products of an image is recyclables.

### Repositories
* Create a directory for the project and initialize git with command `git clone https://github.com/MarkHash/Smart-Recycling-using-Deep-Learning.git`

### Environment Set up
* Download and install python if you don’t have it already.
* Install required packages by command `pip install -r requirements.txt`

### Data
* `app.py`: Web app that classfies recyclables with the model from training data
* `fil_uploader.html`: Simple html code to test the classification
* `222355509_assignment2_solution.ipynb`: The code explored and developed classification model
* `final_model.h5`: The output model from `222355509_assignment2_solution.ipynb`
* `kaggle.json`: API key to download dataset from kaggle
* `dataset-resized.zip`: Download from https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE?resourcekey=0-F-D8v2tnSfByG6ll3t9JxA and it was referred by https://github.com/garythung/trashnet

### Deployment
* Run a command `uvicorn app:app` to deploy locally
* Access `file_uploader.html` in a browser and upload a file to test the classfication
