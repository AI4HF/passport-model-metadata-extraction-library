import sys
import os

from lib.ai4hf_passport_models import LearningDataset, DatasetTransformation, DatasetTransformationStep

# Add 'lib' directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))

from ai4hf_passport_keras import KerasMetadataCollectionAPI
from ai4hf_passport_models import LearningStage, EvaluationMeasure, Model, LearningStageType, EvaluationMeasureType
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example usage of keras library
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Construct an api client for interacting with AI4HF passport server
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

learning_dataset = LearningDataset(datasetId="initial_dataset",
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

# Call this function with your model object
api_client.submit_results_to_ai4hf_passport(model, learning_stages, evaluation_measures, model_info, learning_dataset,
                                            dataset_transformation, dataset_transformation_steps)