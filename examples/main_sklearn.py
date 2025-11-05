import sys
import os
import io
import matplotlib.pyplot as plt
import base64
from lib.ai4hf_passport_models import LearningDataset, DatasetTransformation, DatasetTransformationStep, ModelFigure

# Add 'lib' directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))

from ai4hf_passport_sklearn import SKLearnMetadataCollectionAPI
from ai4hf_passport_models import LearningStage, EvaluationMeasure, Model, LearningStageType, EvaluationMeasureType
import pandas as pd

# Example usage of sklearn library
dataset = pd.read_csv('../test-data/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Construct an api client for interacting with AI4HF passport server
api_client = SKLearnMetadataCollectionAPI(
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

learning_dataset = LearningDataset(datasetId="0197a6fa-6507-775b-99d9-f8808e10052d",
                                   description="Finalized learning dataset for HF Risk Prediction Model Teaching")
dataset_transformation = DatasetTransformation(title="Dataset Smoothening and Normalization",
                                               description="Dataset is transformed by smoothening and normalization.")
dataset_transformation_steps = [
    DatasetTransformationStep(
        inputFeatures="feature1",
        outputFeatures="feature1_1",
        method="Normalization",
        explanation="Decimal values are normalized between 0 and 1.")
]

# Example accuracy values across training epochs
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy = [0.55, 0.60, 0.63, 0.67, 0.70, 0.74, 0.77, 0.79, 0.82, 0.85]

# Create the plot
fig, ax = plt.subplots()
ax.plot(epochs, accuracy, marker='o', linestyle='-', color='b')

# Add title, labels, and grid
ax.set_title("Model Accuracy Over Epochs", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# Save the figure to a BytesIO buffer and convert it to Base64
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
image_bytes = buf.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Define a model figure object for this figure
model_figures = [
    ModelFigure(imageBase64=image_b64)
]

# Call this function with your model object
api_client.submit_results_to_ai4hf_passport(model, learning_stages, evaluation_measures, model_info, learning_dataset,
                                            dataset_transformation, dataset_transformation_steps, model_figures)
