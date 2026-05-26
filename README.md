# MLOps Engineer Assignment

## Assignment
As an MLOps engineer, one of your main goals is to help data scientists bring their ML models production. In this exercise we want to recreate such a scenario.
 
A data scientist wants to create a model that predicts which digit a handwritten number represents. The data scientist has created a Jupyter Notebook that contains code to train a ML model. Within the notebook, it also generates example files that can be used as input to the model. Lastly, there is code to generate a prediction when input is provided to the model.
 
The data scientist wants to serve this model as an application so other people can use the model to identify handwritten numbers. Your task is to productionize the **inference code** provided in the Jupyter Notebook, so that people who want to identify handwritten numbers can call an endpoint and get a prediction.
 
Please pay attention to the required tasks specified below:

## Required Tasks

### 1. Inference API

- Serve the model as a REST API (e.g., FastAPI)
- The API should accept an image and metadata, and return a prediction
- Include input validation and appropriate error handling

### 2. Monitoring & observability: 

- The endpoint should have basic monitoring & observability in place (e.g.: logging, health checks, basic metrics)

### 3. Code Quality & Structure

- Refactor the codebase to improve maintainability
- Add meaningful tests

### 4. Build & Deploy

- Containerize the inference API into a runnable image
- The setup should work out of the box

### 5. CI/CD Pipeline

- Set up a GitHub Actions workflow that:
  - Lints the code
  - Runs the tests
  - Build the solution
- CI/CD process for pull request (Bonus: Automatic release)

### 6. Documentation

- Provide a `README.md` (replace this file) that covers:
  - How to set up, run, and test the project locally
  - How to build the container

## Bonus

- **Cloud deployment**: Provide infrastructure-as code (IaC) that could be used to deploy this application. You may use any IaC language and any services offered by your chosen cloud provider.
- **Model versioning**: implement a scheme to track and serve different model versions

## Guidelines

- Use any libraries or tools you are comfortable with
- Commit your work incrementally with meaningful commit messages (we review git history)

## Submission

- **Clone** or (Prerequisite: Authenticated to GitHub) click **"Use this template"** at the top of the repository to create your own copy 
- Complete the tasks in your newly created repository, then share the link to your repository when you're done.

## FastAPI Conversion

This repository now includes a FastAPI application in `app.py` that exposes a `/predict` endpoint.

### Run locally

1. Create a Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the API

```bash
uvicorn app:app --reload
```

3. Use the endpoint

Send a multipart request to `/predict` with:
- `image`: handwritten digit image file
- `pen_pressure`: float
- `writer_age`: integer
- `handedness`: string (`left` or `right`)

Example response:

```json
{"predicted_digit": 7}
```

### Notes

- The API expects the trained artifacts `image_model.pth`, `final_classifier.pth`, and `metadata_encoder.joblib` to exist in the repository root.
- The notebook training logic can still be used to generate those artifacts before starting the API.
