import requests
import json
import jwt
from typing import Dict, Any
from ai4hf_passport_models import *


class BaseMetadataCollectionAPI:
    """
    Base class for interacting AI4HF Passport Server.
    """

    def __init__(self, passport_server_url: str, study_id: str, experiment_id: str, organization_id: str, connector_secret: str):
        """
        Initialize the API client with authentication and study details.
        """
        self.passport_server_url = passport_server_url
        self.study_id = study_id
        self.experiment_id = experiment_id
        self.organization_id = organization_id
        self.connector_secret = connector_secret
        self.token = self._authenticate()

    def __str__(self):
        return json.dumps({"passport_server_url": self.passport_server_url, "study_id": self.study_id,
                           "experiment_id": self.experiment_id, "organization_id": self.organization_id,
                           "connector_secret": self.connector_secret, "token": self.token})

    def _refreshTokenAndRetry(self, response, headers, payload, url):
        """
        If token is expired, refresh token and retry

        :param response: Response object from previous request.
        :param headers: Headers object from previous request.
        :param payload: Payload object from previous request.
        :param url: The url to sent.

        :return response: Response algorithm object from the server.
        """

        if response.status_code == 401:  # Token expired, refresh and retry
            self.token = self._authenticate()
            headers["Authorization"] = f"Bearer {self.token}"
            return requests.post(url, json=payload, headers=headers)
        else:
            return response

    def _authenticate(self) -> str:
        """
        Authenticate with login endpoint and retrieve an access token.
        """
        auth_url = f"{self.passport_server_url}/user/connector/login"

        response = requests.post(auth_url, data=self.connector_secret)
        response.raise_for_status()
        return response.json().get("access_token")

    def submit_algorithm(self, algorithm: Algorithm) -> Algorithm:
        """
        Submit algorithm to the AI4HF Passport Server.

        :param algorithm: Algorithm object to be sent.

        :return algorithm: Created algorithm object from the server.
        """
        url = f"{self.passport_server_url}/algorithm?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"name": algorithm.name, "objectiveFunction": algorithm.objectiveFunction,
                   "subType": algorithm.subType, "type": algorithm.type}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()
        return Algorithm(response_json.get('name'), response_json.get('objectiveFunction'),
                         response_json.get('type'), response_json.get('subType'),
                         response_json.get('algorithmId'))

    def submit_implementation(self, implementation: Implementation) -> Implementation:
        """
        Submit implementation to the AI4HF Passport Server.

        :param implementation: Implementation object to be sent.

        :return implementation: Created Implementation object from the server.
        """
        url = f"{self.passport_server_url}/implementation?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"algorithmId": implementation.algorithmId, "name": implementation.name,
                   "software": implementation.software, "description": implementation.description}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()
        return Implementation(response_json.get('name'), response_json.get('software'),
                              response_json.get('description'), response_json.get('algorithmId'),
                              response_json.get('implementationId'))

    def submit_learning_process(self, learning_process_info: Dict[str, any]) -> LearningProcess:
        """
        Submit learning process to the AI4HF Passport Server.

        :param learning_process_info: Dictionary object that holds learning process related fields.

        :return response: Created Learning Process object from the server.
        """
        algorithm_dict: dict = learning_process_info['implementation']['algorithm']
        algorithm = self.submit_algorithm(Algorithm(algorithm_dict.get('name'), algorithm_dict.get('objectiveFunction'),
                                                    algorithm_dict.get('type'), algorithm_dict.get('subType'),
                                                    algorithm_dict.get('algorithmId')))

        learning_process_info['implementation']['algorithmId'] = algorithm.algorithmId

        implementation_dict: dict = learning_process_info['implementation']
        implementation = Implementation(implementation_dict.get('name'), implementation_dict.get('software'),
                                        implementation_dict.get('description'), implementation_dict.get('algorithmId'),
                                        implementation_dict.get('implementationId'))
        implemantation_submitted = self.submit_implementation(implementation)

        learning_process = LearningProcess(None, self.study_id,
                                           implemantation_submitted.implementationId,
                                           learning_process_info.get('description', None))

        url = f"{self.passport_server_url}/learning-process?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"implementationId": learning_process.implementationId, "studyId": learning_process.studyId,
                   "description": learning_process.description}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()
        return LearningProcess(response_json.get('learningProcessId'), response_json.get('studyId'),
                               response_json.get('implementationId'), response_json.get('description'))

    def submit_learning_dataset_and_transformation(self, learning_dataset: LearningDataset,
                                                   dataset_transformation: DatasetTransformation) -> (
            LearningDataset, DatasetTransformation):
        """
        Submit LearningDataset and Transformation to the AI4HF Passport Server.

        :param learning_dataset: LearningDataset object to be sent.
        :param dataset_transformation: DatasetTransformation object to be sent.

        :return learning_dataset, dataset_transformation: Created learning_dataset and dataset_transformation objects from the server.
        """
        url = f"{self.passport_server_url}/learning-dataset?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        dataset_transformation_json = {
            "description": dataset_transformation.description,
            "title": dataset_transformation.title
        }
        learning_dataset_json = {
            "datasetId": learning_dataset.datasetId,
            "description": learning_dataset.description,
        }
        payload = {"datasetTransformation": dataset_transformation_json,
                   "learningDataset": learning_dataset_json}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()

        response_learning_dataset = response_json.get('learningDataset')
        response_dataset_transformation = response_json.get('datasetTransformation')

        return_learning_dataset = LearningDataset(datasetId=response_learning_dataset.get('datasetId'),
                                                  studyId=response_learning_dataset.get('studyId'),
                                                  description=response_learning_dataset.get('description'),
                                                  dataTransformationId=response_learning_dataset.get(
                                                      'dataTransformationId'),
                                                  learningDatasetId=response_learning_dataset.get('learningDatasetId'))

        return_dataset_transformation = DatasetTransformation(title=response_dataset_transformation.get('title'),
                                                              description=response_dataset_transformation.get(
                                                                  'description'),
                                                              dataTransformationId=response_dataset_transformation.get(
                                                                  'dataTransformationId'))
        return return_learning_dataset, return_dataset_transformation

    def submit_dataset_transformation_step(self,
                                           dataset_transformation_step: DatasetTransformationStep) -> DatasetTransformationStep:
        """
        Submit DatasetTransformationStep to the AI4HF Passport Server.

        :param dataset_transformation_step: DatasetTransformationStep object to be sent.

        :return response: Created DatasetTransformationStep object from the server.
        """

        url = f"{self.passport_server_url}/dataset-transformation-step?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"createdBy": dataset_transformation_step.createdBy,
                   "dataTransformationId": dataset_transformation_step.dataTransformationId,
                   "explanation": dataset_transformation_step.explanation,
                   "inputFeatures": dataset_transformation_step.inputFeatures,
                   "lastUpdatedBy": dataset_transformation_step.lastUpdatedBy,
                   "method": dataset_transformation_step.method,
                   "outputFeatures": dataset_transformation_step.outputFeatures}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()
        return DatasetTransformationStep(dataTransformationId=response_json.get('dataTransformationId'),
                                         inputFeatures=response_json.get('inputFeatures'),
                                         outputFeatures=response_json.get('outputFeatures'),
                                         method=response_json.get('method'),
                                         explanation=response_json.get('explanation'),
                                         createdBy=response_json.get('createdBy'),
                                         lastUpdatedBy=response_json.get('lastUpdatedBy'),
                                         createdAt=response_json.get('createdAt'),
                                         lastUpdatedAt=response_json.get('lastUpdatedAt'),
                                         stepId=response_json.get('stepId'))

    def extract_learning_process(self, model: Any) -> Dict[str, Any]:
        """
        This method must be overridden in derived class. Implement library specific extraction of learning process.

        :param model: Model class that specific to implemented library.

        :return dict: Extracted learning process fields.
        """
        pass

    def extract_and_submit_learning_process(self, model: Any):
        """
        Extract and submit learning process into the server.

        :param model: Model class that specific to implemented library.

        :return response: Response json from the server.
        """
        learning_process = self.extract_learning_process(model)
        return self.submit_learning_process(learning_process)

    def submit_learning_stage(self, learning_stage: LearningStage):
        """
        Submit learning stage to the AI4HF Passport Server.

        :param learning_stage: LearningStage object to be sent.

        :return response: The response of the server for creating the learning stage.
        """

        url = f"{self.passport_server_url}/learning-stage?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"learningProcessId": learning_stage.learningProcessId,
                   "learningStageName": learning_stage.learningStageName,
                   "description": learning_stage.description, "datasetPercentage": learning_stage.datasetPercentage}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        return response.json()

    def submit_model(self, model: Model) -> Model:
        """
        Submit model to the AI4HF Passport Server.

        :param model: Model object that will be sent to the server.

        :return response: The response of the server for creating the model.
        """

        url = f"{self.passport_server_url}/model?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "learningProcessId": model.learningProcessId,
            "studyId": model.studyId,
            "experimentId": model.experimentId,
            "name": model.name,
            "version": model.version,
            "tag": model.tag,
            "modelType": model.modelType,
            "productIdentifier": model.productIdentifier,
            "owner": model.owner,
            "trlLevel": model.trlLevel,
            "license": model.license,
            "primaryUse": model.primaryUse,
            "secondaryUse": model.secondaryUse,
            "intendedUsers": model.intendedUsers,
            "counterIndications": model.counterIndications,
            "ethicalConsiderations": model.ethicalConsiderations,
            "limitations": model.limitations,
            "fairnessConstraints": model.fairnessConstraints,
            "createdBy": model.createdBy,
            "lastUpdatedBy": model.lastUpdatedBy}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json = response.json()
        return Model(
            version=response_json.get('version'),
            tag=response_json.get('tag'),
            modelType=response_json.get('modelType'),
            productIdentifier=response_json.get('productIdentifier'),
            trlLevel=response_json.get('trlLevel'),
            license=response_json.get('license'),
            primaryUse=response_json.get('primaryUse'),
            secondaryUse=response_json.get('secondaryUse'),
            intendedUsers=response_json.get('intendedUsers'),
            counterIndications=response_json.get('counterIndications'),
            ethicalConsiderations=response_json.get('ethicalConsiderations'),
            limitations=response_json.get('limitations'),
            fairnessConstraints=response_json.get('fairnessConstraints'),
            createdBy=response_json.get('createdBy'),
            lastUpdatedBy=response_json.get('lastUpdatedBy'),
            modelId=response_json.get('modelId'),
            learningProcessId=response_json.get('learningProcessId'),
            studyId=response_json.get('studyId'),
            experimentId=response_json.get('experimentId'),
            name=response_json.get('name'),
            owner=response_json.get('owner')
        )

    def extract_model(self,
                      model: Any,
                      model_info: Model) -> Model:
        """
        This method must be overridden in derived class. Implement library specific extraction of model.

        :param model: Model class that specific to implemented library.
        :param model_info: Model object for model related fields.

        :return dict: Extracted model object.
        """
        pass

    def extract_and_submit_model(self,
                                 model: Any,
                                 model_info: Model) -> Model:
        """
        Extract and submit model into the server.

        :param model: Model class that specific to implemented library.
        :param model_info: Model object for model related fields.

        :return created_model: Created model object.
        """
        extracted_model: Model = self.extract_model(model, model_info)
        return self.submit_model(extracted_model)

    def submit_evaluation_measure(self, evaluation_measure: EvaluationMeasure):
        """
        Submit evaluation measure to the AI4HF Passport Server.

        :param evaluation_measure: Dictionary object that holds evaluation measure related fields.

        :return response: The response of the server for creating the evaluation measure.
        """

        url = f"{self.passport_server_url}/evaluation-measure?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "modelId": evaluation_measure.modelId,
            "name": evaluation_measure.name,
            "value": evaluation_measure.value,
            "dataType": evaluation_measure.dataType,
            "description": evaluation_measure.description
        }

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        return response.json()

    def submit_results_to_ai4hf_passport(self,
                                         model: Any,
                                         learning_stages: list[LearningStage],
                                         evaluation_measures: list[EvaluationMeasure],
                                         model_info: Model,
                                         learning_dataset: LearningDataset,
                                         dataset_transformation: DatasetTransformation,
                                         dataset_transformation_steps: list[DatasetTransformationStep]):
        """
        Submit results of the ML model to the AI4HF Passport Server.

        :param model: Model class that specific to implemented library.
        :param learning_stages: The list of learning stages.
        :param evaluation_measures: The list of evaluation measures.
        :param model_info: Model class for model related fields.

        """
        print('Sending given informations into AI4HF Passport server....')
        learning_process = self.extract_and_submit_learning_process(model)
        print(f'Learning process created: {learning_process}')

        for learning_stage in learning_stages:
            learning_stage.learningProcessId = learning_process.learningProcessId
            learning_stage_response = self.submit_learning_stage(learning_stage)
            print(f'Learning stage created: {learning_stage_response}')

        user_id = jwt.decode(self.token, options={"verify_signature": False})['sub']
        model_info.learningProcessId = learning_process.learningProcessId
        model_info.studyId = self.study_id
        model_info.experimentId = self.experiment_id
        model_info.owner = self.organization_id
        model_info.createdBy = user_id
        model_info.lastUpdatedBy = user_id
        created_model: Model = self.extract_and_submit_model(model, model_info)
        print(f'Model created: {created_model}')

        for evaluation_measure in evaluation_measures:
            evaluation_measure.modelId = created_model.modelId
            evaluation_measure_response = self.submit_evaluation_measure(evaluation_measure)
            print(f'Evaluation measure created: {evaluation_measure_response}')

        self.extract_and_submit_parameters(model, created_model.modelId)

        learning_dataset.studyId = self.study_id
        response_learning_dataset, response_dataset_transformation = self.submit_learning_dataset_and_transformation(
            learning_dataset=learning_dataset,
            dataset_transformation=dataset_transformation)
        print(f'Learning Dataset created: {response_learning_dataset}')
        print(f'Dataset Transformation created: {response_dataset_transformation}')

        for dataset_transformation_step in dataset_transformation_steps:
            dataset_transformation_step.dataTransformationId = response_dataset_transformation.dataTransformationId
            dataset_transformation_step.createdBy = user_id
            dataset_transformation_step.lastUpdatedBy = user_id
            response_dataset_transformation_step = self.submit_dataset_transformation_step(
                dataset_transformation_step=dataset_transformation_step)
            print(f'Dataset Transformation Step created: {response_dataset_transformation_step}')

        print('Given informations are sent into AI4HF Passport server!')
        pass

    def submit_parameter(self, parameter: Parameter) -> Parameter:
        """
        Submit parameter to the AI4HF Passport Server.

        :param parameter: Parameter object to be sent.

        :return parameter: Created parameter object from the server.
        """
        url = f"{self.passport_server_url}/parameter?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"studyId": parameter.studyId, "name": parameter.name,
                   "dataType": parameter.dataType, "description": parameter.description}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return Parameter(response_json.get('parameterId'), response_json.get('studyId'),
                         response_json.get('name'), response_json.get('dataType'),
                         response_json.get('description'))

    def submit_model_parameter(self, model_parameter: ModelParameter) -> ModelParameter:
        """
        Submit ModelParameter to the AI4HF Passport Server.

        :param model_parameter: ModelParameter object to be sent.

        :return ModelParameter: Created ModelParameter object from the server.
        """
        url = f"{self.passport_server_url}/model-parameter?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"modelId": model_parameter.modelId, "parameterId": model_parameter.parameterId,
                   "type": model_parameter.type, "value": model_parameter.value}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return ModelParameter(response_json.get('id', {}).get('modelId'),
                              response_json.get('id', {}).get('parameterId'), response_json.get('type'),
                              response_json.get('value'))

    def extract_parameters(self,
                           model: Any) -> list[Dict[str, Any]]:
        """
        This method must be overridden in derived class. Implement library specific extraction of parameters.

        :param model: Model class that specific to implemented library.

        :return list[dict]: Extracted parameters.
        """
        pass

    def extract_and_submit_parameters(self,
                                      model: Any, model_id: str):
        """
        Extract and submit parameters into the server.

        :param model: Model class that specific to implemented library.
        :param model_id: The ID of the model.

        :return response: Response json from the server.
        """
        extracted_parameters = self.extract_parameters(model)
        for extracted_parameter_dict in extracted_parameters:
            extracted_parameter = Parameter(extracted_parameter_dict.get('parameterId'),
                                            extracted_parameter_dict.get('studyId'),
                                            extracted_parameter_dict.get('name'),
                                            extracted_parameter_dict.get('dataType'),
                                            extracted_parameter_dict.get('description'))
            created_parameter = self.submit_parameter(extracted_parameter)
            model_parameter = ModelParameter(model_id, created_parameter.parameterId,
                                             extracted_parameter_dict.get('type', None),
                                             extracted_parameter_dict.get('value', None))
            created_model_parameter = self.submit_model_parameter(model_parameter)
            print(f'Model Parameter created: {created_model_parameter}')

        pass
