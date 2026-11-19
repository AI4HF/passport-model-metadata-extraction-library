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
pip install pandas scikit-learn torch tensorflow requests matplotlib
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
Behind the proxy, the Passport UI is at http://localhost/ai4hf/passport and the API is at
http://localhost/ai4hf/passport/api.

After completing your machine learning work, provide the trained model object to this library. The library
automatically extracts essential metadata from the model; the rest is supplied by the developer. Run the code
below after your machine learning pipeline. For full examples, see `examples/main_sklearn.py` (every option),
`examples/main_keras.py` and `examples/main_torch.py`.

```python
import sys; sys.path.insert(0, "path/to/passport-model-metadata-extraction-library/lib")   # not pip-installable yet

from ai4hf_passport_sklearn import SKLearnMetadataCollectionAPI   # or KerasMetadataCollectionAPI / TorchMetadataCollectionAPI
from ai4hf_passport_models import Model, ModelEvaluation, EvaluationMeasure, EvaluationMeasureType, LearningDataset

passport = SKLearnMetadataCollectionAPI(
    passport_server_url="http://localhost:80/ai4hf/passport/api",
    keycloak_server_url="http://localhost:8081",
    client_id="ai4hf-fl-central", client_secret="<client secret>",   # Keycloak service account of this library
    study_id="<study id>", experiment_id="<experiment id>", organization_id="<owner organization id>",
)

created = passport.submit_results_to_ai4hf_passport(
    model=model,                                                     # your trained model object
    model_info=Model(name="My model", version="1.0"),
    model_evaluation=ModelEvaluation(trigger="training", description="Evaluation on the test split."),
    evaluation_measures=[EvaluationMeasure(EvaluationMeasureType.ACCURACY, 0.91)],
    learning_dataset=LearningDataset(datasetId="<dataset id>"),     # a dataset registered in the Passport
)
```

These five arguments are required. The optional ones add detail to the passport:
- `learning_stages`
- `dataset_transformation` and `dataset_transformation_steps`
- `learning_process_parameters` and `learning_stage_parameters`; these appear in the passport
- `model_figures`, built with `ModelFigure.from_matplotlib(fig, title=...)`

For a model trained across several sites:
- Pass a list with one `LearningDataset(datasetId=..., weight=<records>)` per site.
- Record each site's own evaluation with `submit_evaluation_run`.

The model's hyperparameters are stored too, but they are not shown in the passport. Learned weights are only sent
with `include_learned_parameters=True`. The call returns the created ids (`modelId`, ...). A
QUALITY_ASSURANCE_SPECIALIST then generates the model's passport in the Passport UI (Passport Management).

The library logs in as a Keycloak service account. The deployment ships the `ai4hf-fl-central` client for it,
a DATA_SCIENTIST in both demo studies.

The examples write into the demo study of the deployment. Their connection settings can be overridden with the
`PASSPORT_SERVER_URL`, `KEYCLOAK_SERVER_URL`, `CLIENT_ID` and `CLIENT_SECRET` environment variables.
