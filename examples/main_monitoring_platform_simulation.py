"""
Demo data for the passport-monitoring-platform-connector: submits a series of models whose metrics drift a
little from one model to the next (each model becomes one round on the monitoring dashboard).

NUMBER_OF_MODELS (environment variable, default 20) sets how many models are submitted.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # library is not pip-installable yet

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ai4hf_passport_sklearn import SKLearnMetadataCollectionAPI
from ai4hf_passport_models import Model, ModelEvaluation, EvaluationMeasure, EvaluationMeasureType, LearningDataset

NUMBER_OF_MODELS = int(os.getenv("NUMBER_OF_MODELS", "20"))

model = DecisionTreeClassifier(max_depth=3).fit(np.random.rand(100, 2), np.random.randint(0, 2, 100))

passport = SKLearnMetadataCollectionAPI(
    passport_server_url=os.getenv("PASSPORT_SERVER_URL", "http://localhost:80/ai4hf/passport/api"),
    keycloak_server_url=os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8081"),
    client_id=os.getenv("CLIENT_ID", "ai4hf-fl-central"),
    client_secret=os.getenv("CLIENT_SECRET", "fT6yH2nQ9wE4rP7xK1mZ5vB8cJ3dL0sA"),  # demo secret of the local deployment
    study_id="0197a6f8-2b78-71e4-81c1-b7b6a744ece3",
    experiment_id="0197a6f9-1f49-74a5-ab8a-e64fae0ca141",
    organization_id="0197a6f5-bb48-7855-b248-95697e913f22",
)

metric_types = [EvaluationMeasureType.PRECISION, EvaluationMeasureType.SENSITIVITY,
                EvaluationMeasureType.RECALL, EvaluationMeasureType.F1_SCORE]
values = {t: random.uniform(0.5, 0.9) for t in metric_types}

for i in range(NUMBER_OF_MODELS):
    values = {t: min(v + random.uniform(-0.03, 0.06), 0.99) for t, v in values.items()}  # drift
    created = passport.submit_results_to_ai4hf_passport(
        model=model,
        model_info=Model(name=f"model_{i}", version=f"{i}.0", tag="monitoring-simulation"),
        model_evaluation=ModelEvaluation(trigger="training", description=f"Simulated evaluation of model {i}."),
        evaluation_measures=[EvaluationMeasure(t, v) for t, v in values.items()],
        learning_dataset=LearningDataset(datasetId="0197a6fa-6507-775b-99d9-f8808e10052d",
                                         description="Learning dataset of the monitoring simulation"),
    )
    print(f"Model {i + 1}/{NUMBER_OF_MODELS} created: {created['modelId']}")
