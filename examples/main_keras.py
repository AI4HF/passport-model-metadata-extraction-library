"""
Register a Keras model in the AI4HF Passport, with only the required arguments.

Identical to the scikit-learn case apart from the client class: KerasMetadataCollectionAPI reads the
architecture, optimizer and loss from the model. See main_sklearn.py for all optional arguments.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # library is not pip-installable yet

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from ai4hf_passport_keras import KerasMetadataCollectionAPI
from ai4hf_passport_models import Model, ModelEvaluation, EvaluationMeasure, EvaluationMeasureType, LearningDataset

# 1. Your training code (a stand-in on random data)
model = Sequential([Input(shape=(2,)), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.random.rand(100, 2), np.random.randint(0, 2, 100), epochs=3, verbose=0)

# 2. Connect to the Passport
passport = KerasMetadataCollectionAPI(
    passport_server_url=os.getenv("PASSPORT_SERVER_URL", "http://localhost:80/ai4hf/passport/api"),
    keycloak_server_url=os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8081"),
    client_id=os.getenv("CLIENT_ID", "ai4hf-fl-central"),
    client_secret=os.getenv("CLIENT_SECRET", "fT6yH2nQ9wE4rP7xK1mZ5vB8cJ3dL0sA"),  # demo secret of the local deployment
    study_id="0197a6f8-2b78-71e4-81c1-b7b6a744ece3",
    experiment_id="0197a6f9-1f49-74a5-ab8a-e64fae0ca141",
    organization_id="0197a6f5-bb48-7855-b248-95697e913f22",
)

# 3. Describe the training and submit it
created = passport.submit_results_to_ai4hf_passport(
    model=model,
    model_info=Model(name="Purchase prediction (Keras)", version="1.0"),
    model_evaluation=ModelEvaluation(trigger="training", description="Evaluation on the test split."),
    evaluation_measures=[EvaluationMeasure(EvaluationMeasureType.ACCURACY, 0.88),
                         EvaluationMeasure(EvaluationMeasureType.AUC, 0.93)],
    learning_dataset=LearningDataset(datasetId="0197a6fa-6507-775b-99d9-f8808e10052d",
                                     description="Purchase records used for training"),
)

print(f"Done. The passport of model {created['modelId']} can now be generated in the Passport UI.")
