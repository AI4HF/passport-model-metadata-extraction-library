"""
Register a scikit-learn model in the AI4HF Passport, using every option of the library.

Step 1 is ordinary training code. Steps 2-3 are what you add after your own training. Only model, model_info,
model_evaluation, evaluation_measures and learning_dataset are required; everything marked "optional" adds
more detail to the passport.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # library is not pip-installable yet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ai4hf_passport_sklearn import SKLearnMetadataCollectionAPI
from ai4hf_passport_models import (Model, ModelEvaluation, EvaluationMeasure, EvaluationMeasureType, LearningDataset,
                                   LearningStage, LearningStageType, LearningProcessParameter, LearningStageParameter,
                                   DatasetTransformation, DatasetTransformationStep, ModelFigure)

# ---------------------------------------------------------------------------------------------------------------
# 1. Your training code
# ---------------------------------------------------------------------------------------------------------------
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-data", "Social_Network_Ads.csv"))
X_train, X_test, y_train, y_test = train_test_split(data[["Age", "EstimatedSalary"]], data["Purchased"],
                                                    test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_train)
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0).fit(scaler.transform(X_train), y_train)
y_score = model.predict_proba(scaler.transform(X_test))[:, 1]
y_pred = (y_score >= 0.5).astype(int)

# ---------------------------------------------------------------------------------------------------------------
# 2. Connect to the Passport
# ---------------------------------------------------------------------------------------------------------------
passport = SKLearnMetadataCollectionAPI(
    passport_server_url=os.getenv("PASSPORT_SERVER_URL", "http://localhost:80/ai4hf/passport/api"),
    keycloak_server_url=os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8081"),
    client_id=os.getenv("CLIENT_ID", "ai4hf-fl-central"),
    client_secret=os.getenv("CLIENT_SECRET", "fT6yH2nQ9wE4rP7xK1mZ5vB8cJ3dL0sA"),  # demo secret of the local deployment
    study_id="0197a6f8-2b78-71e4-81c1-b7b6a744ece3",        # the study, experiment and organization
    experiment_id="0197a6f9-1f49-74a5-ab8a-e64fae0ca141",   # must already exist in the Passport
    organization_id="0197a6f5-bb48-7855-b248-95697e913f22", # organization that owns the model
)

# ---------------------------------------------------------------------------------------------------------------
# 3. Describe the training and submit it
# ---------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)

created = passport.submit_results_to_ai4hf_passport(
    model=model,

    # The model's descriptive fields (name, version, license, intended use, limitations, ...)
    model_info=Model(name="Purchase prediction", version="1.0", license="Apache-2.0",
                     intendedUsers="Developers trying out the AI4HF Passport",
                     limitations="Toy dataset; not for clinical use."),

    # The evaluation and its metrics
    model_evaluation=ModelEvaluation(trigger="training", description="Evaluation on the held-out test split."),
    evaluation_measures=[
        EvaluationMeasure(EvaluationMeasureType.ACCURACY, accuracy_score(y_test, y_pred)),
        EvaluationMeasure(EvaluationMeasureType.F1_SCORE, f1_score(y_test, y_pred)),
        EvaluationMeasure(EvaluationMeasureType.AUC, roc_auc_score(y_test, y_score)),
    ],

    # The dataset (registered in the Passport) the model was trained on
    learning_dataset=LearningDataset(datasetId="0197a6fa-6507-775b-99d9-f8808e10052d",
                                     description="Purchase records used for training", weight=len(data)),

    # Optional: how the dataset was prepared, step by step
    dataset_transformation=DatasetTransformation(title="Standardization",
                                                 description="Numeric features are standardized."),
    dataset_transformation_steps=[
        DatasetTransformationStep(inputFeatures="Age", outputFeatures="Age", method="StandardScaler",
                                  explanation="Scaled to mean 0 and unit variance."),
        DatasetTransformationStep(inputFeatures="EstimatedSalary", outputFeatures="EstimatedSalary",
                                  method="StandardScaler", explanation="Scaled to mean 0 and unit variance."),
    ],

    # Optional: the training stages and the share of the data each used
    learning_stages=[LearningStage(LearningStageType.TRAINING, datasetPercentage=70),
                     LearningStage(LearningStageType.TEST, datasetPercentage=30)],

    # Optional: settings that should appear in the passport
    learning_process_parameters=[LearningProcessParameter("feature_scaling", value="StandardScaler")],
    learning_stage_parameters=[LearningStageParameter("split_random_state", value=0, stage=LearningStageType.TRAINING)],

    # Optional: figures
    model_figures=[ModelFigure.from_matplotlib(fig, title="ROC curve", description="ROC curve on the test split.")],
)

print(f"Done. The passport of model {created['modelId']} can now be generated in the Passport UI.")
