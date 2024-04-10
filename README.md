# Project Readme

This repository contains starter code for the **Census Income Prediction** project as part of the Udacity Data Scientist Nanodegree program.

## Overview

The project aims to develop a machine learning model to predict whether an individual's income exceeds $50,000 per year based on census data. The project involves several stages including data preprocessing, model development, API creation, and deployment.

## Environment Setup

To set up your environment for this project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/geremyb12/nd0821-c3-starter-code.git
Install Dependencies: Set up a conda environment with the necessary dependencies by running:

conda create -n [envname] "python=3.8"

conda activate [envname]

pip install src/requirements.txt

## GitHub Actions Setup
To set up GitHub Actions for this project, follow these steps:

Configure GitHub Actions on your repository to run pytest and flake8 on push.
## Data Preparation
Download the census.csv dataset from the data folder in the starter repository.
Clean the data by removing spaces.
## Model Development
Write a machine learning model using the starter code provided.
Write unit tests for at least 3 functions in the model code.
Write a function to output the performance of the model on slices of the data.
Write a model card using the provided template.
## Evaluation Model Metrics
Evaluation on Test Set:
Precision: 0.7093596059113301
Recall: 0.2755102040816326
Fbeta: 0.3968764354616444
## API Creation
Create a RESTful API using FastAPI with endpoints for model inference.
Write unit tests to test the API endpoints.
API Deployment
Deploy the API using Render:

Free Instance: Set up a free instance on Render.
Configure Render: Configure Render to deploy your API.
GitHub Actions Integration: Integrate Render with GitHub Actions for continuous deployment.
Running the Application
To run the application locally, follow these steps:

Start the FastAPI server using the following command:

bash
Copy code
uvicorn src.main:app --host 0.0.0.0 --port 10000
Access the API at http://localhost:10000.

## Scripts Overview
main.py: Contains the FastAPI app with endpoints for model inference.
model.py: Defines functions for training, evaluating, and inference with the machine learning model.
train_model.py: Script to train the machine learning model, evaluate its performance, and save it.
requirements.txt: List of dependencies for the project.
### Test Scripts
The repository also contains two test scripts under the test/ directory:

test_main.py: Contains unit tests for the API endpoints.
test_model.py: Contains unit tests for the machine learning model.
## Contributors
Geremy Bantug

Feel free to contribute to this project by forking the repository and submitting pull requests with your enhancements or fixes.

For any questions or issues, please open an issue in the repository.