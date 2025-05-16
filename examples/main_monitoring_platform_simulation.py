import sys
import os
import random

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

# Evaluation measure types for the monitoring platform simulation
evaluation_measure_types = [EvaluationMeasureType.PRECISION, EvaluationMeasureType.SENSITIVITY,
                            EvaluationMeasureType.RECALL, EvaluationMeasureType.F1_SCORE]

def get_biased_random_number(threshold: float) -> float:
    r = random.random()  # value between 0.0 and 1.0
    return threshold if r < threshold else r

def get_random_int(min_val: int, max_val: int) -> int:
    return random.randint(min_val, max_val)  # inclusive of both ends


number_of_models = 20
arr = [[] for _ in range(len(evaluation_measure_types))]
for index, evaluation_measure_type in enumerate(evaluation_measure_types):
    base_value = get_biased_random_number(0.5)
    for i in range(number_of_models):
        increment = random.random() * 0.06
        decrement = random.random() * 0.03

        final_value = min(base_value + increment - decrement, 0.99)

        arr[index].append(EvaluationMeasure(evaluation_measure_type, final_value))

for i in range(number_of_models):
    evaluation_measure_for_model = []
    for index, _ in enumerate(evaluation_measure_types):
        evaluation_measure_for_model.append(arr[index][i])

    model_info = Model(
        name = f"model_{i}"
    )

    api_client.submit_results_to_ai4hf_passport(classifier, learning_stages, evaluation_measure_for_model, model_info)
