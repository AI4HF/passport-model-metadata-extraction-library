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
        study_id="initial_study",
        experiment_id="initial_experiment",
        organization_id="initial_organization",
        username="data_scientist",
        password="data_scientist"
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
