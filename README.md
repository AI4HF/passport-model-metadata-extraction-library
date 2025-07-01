# Machine Learning Metadata Extraction Library

<div align="center" style="background-color: white">
  <a href="https://www.ai4hf.com/">
    <img height="60px" src="assets/ai4hf_logo.svg" alt="AI4HF Project"/>
  </a>
</div>

<br/>

<p align="center">
  <a href="https://github.com/DataTools4Heart/common-data-model">
    <img src="https://img.shields.io/github/license/DataTools4Heart/common-data-model" alt="License">
  </a>
  <a href="https://img.shields.io/github/license/DataTools4Heart/releases">
    <img src="https://img.shields.io/github/v/release/DataTools4Heart/common-data-model" alt="Releases">
  </a>
</p>

<br/>


This library provides a unified way to extract metadata from machine learning models built using Scikit-learn, TensorFlow/Keras, and PyTorch. It also includes a base API client to facilitate interaction with AI4HF Passport Server. Extracted metadata can be sent to AI4HF Passport Server.

## Usage
Install necessary dependencies:
```
pip install pandas scikit-learn torch tensorflow requests pyjwt
```
Deploy the Passport server before running the API client. 
Clone passport server repository:
```
git clone https://github.com/AI4HF/passport.git
```
To deploy the Passport server and a proxy:
```
sh passport/docker/deployment/run.sh
sh passport/docker/proxy/run.sh
```
After completing your machine learning work, provide the trained model object to this library. The library will automatically extract essential metadata from the model, while additional metadata must be supplied by the developer. Ensure that the `learning_stages`, `evaluation_measures`, and `model_info` variables are created as shown in the example below. This code snippet should be executed after your machine learning pipeline. For full examples, refer to the `main_keras.py`, `main_sklearn.py`, and `main_torch.py` files.
```python
# Construct an api client(you can choose one of them: KerasMetadaCollectionAPI, TorchMetadataCollectionAPI or SKLearnMetadataCollectionAPI) for interacting with AI4HF passport server
api_client = KerasMetadataCollectionAPI(
    passport_server_url="http://localhost:80/ai4hf/passport/api",
    study_id="0197a6f8-2b78-71e4-81c1-b7b6a744ece3",
    experiment_id="0197a6f9-1f49-74a5-ab8a-e64fae0ca141",
    organization_id="0197a6f5-bb48-7855-b248-95697e913f22",
    connector_secret="eyJhbGciOiJIUzUxMiIsInR5cCIgOiAiSldUIiwi"
                     "a2lkIiA6ICI5ZTFiZTExNi0yMzg1LTRlZDctYTBi"
                     "OC01ZDc0NWNjYzllOGMifQ.eyJpYXQiOjE3NTEyN"
                     "zA4MjgsImp0aSI6ImIxMWE5NGI1LWQ5MzItNDhiN"
                     "C1iMjc4LWFkZjQ1ZDJjMTMxOCIsImlzcyI6Imh0d"
                     "HA6Ly9rZXljbG9hazo4MDgwL3JlYWxtcy9BSTRIR"
                     "i1BdXRob3JpemF0aW9uIiwiYXVkIjoiaHR0cDovL"
                     "2tleWNsb2FrOjgwODAvcmVhbG1zL0FJNEhGLUF1d"
                     "Ghvcml6YXRpb24iLCJzdWIiOiJkYXRhX3NjaWVud"
                     "GlzdCIsInR5cCI6Ik9mZmxpbmUiLCJhenAiOiJBS"
                     "TRIRi1BdXRoIiwic2Vzc2lvbl9zdGF0ZSI6IjE3Y"
                     "zU2ZjhkLTljZmEtNDM2OC05MzQ4LTkzN2ZjY2QyM"
                     "jY0ZCIsInNjb3BlIjoib2ZmbGluZV9hY2Nlc3Mgc"
                     "HJvZmlsZSBlbWFpbCIsInNpZCI6IjE3YzU2ZjhkL"
                     "TljZmEtNDM2OC05MzQ4LTkzN2ZjY2QyMjY0ZCJ9."
                     "obYaa744bmJoQAFO-nh1sCwPKwArWaOUo9_a1I0U"
                     "zc--HBuTLy6oOJVmnVI62bxnMkqoYo97SYGlKGKw"
                     "VStz5g"
    )

# Provide learning stages
learning_stages = [
    LearningStage(learningStageType = LearningStageType.TRAINING,
                  datasetPercentage = 70),
    LearningStage(learningStageType = LearningStageType.TEST,
                  datasetPercentage = 30)
]

# Provide evaluation measures
evaluation_measures = [
    EvaluationMeasure(EvaluationMeasureType.ACCURACY,
                      value = "0.94"),
    EvaluationMeasure(EvaluationMeasureType.F1_SCORE,
                      value = "0.88")
]

# Provide model details
model_info = Model(
    name = "test"
)

# Call this function with your model object
api_client.submit_results_to_ai4hf_passport(model, learning_stages, evaluation_measures, model_info)
```
