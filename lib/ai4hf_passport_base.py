import copy
import requests
import json
import warnings
from typing import Dict, Any, Optional, Union
from ai4hf_passport_models import *

class BaseMetadataCollectionAPI:
    """
    Base class for interacting AI4HF Passport Server.
    """

    def __init__(self, passport_server_url: str, study_id: str, experiment_id: str, organization_id: str,
                 keycloak_server_url: str, client_id: str, client_secret: str,
                 keycloak_realm: str = "AI4HF-Authorization",
                 include_learned_parameters: bool = False):
        """
        Initialize the API client with authentication and study details.

        The library authenticates as a Keycloak *service account* using the client_credentials grant.
        The service account must be a member of the study group with the DATA_SCIENTIST role, exactly
        as a person would be, and the Passport resolves the token's azp claim to a SoftwareAgent so
        the audit log book records which component wrote what.

        :param include_learned_parameters: Also submit learned values (weights, biases, coefficients,
            feature importances, ...) as model parameters. Off by default: they mean one Passport request
            pair per array and are not human-readable in a passport. Hyperparameters are always submitted.
        """
        self.passport_server_url = passport_server_url.rstrip("/")
        self.study_id = study_id
        self.experiment_id = experiment_id
        self.organization_id = organization_id
        self.keycloak_server_url = keycloak_server_url.rstrip("/")
        self.keycloak_realm = keycloak_realm
        self.client_id = client_id
        self.client_secret = client_secret
        self.include_learned_parameters = include_learned_parameters
        self._parameter_cache: Optional[Dict[str, Parameter]] = None
        self.token = self._authenticate()

    def __str__(self):
        return json.dumps({"passport_server_url": self.passport_server_url, "study_id": self.study_id,
                           "experiment_id": self.experiment_id, "organization_id": self.organization_id,
                           "client_id": self.client_id})

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        GET a Passport endpoint, re-authenticating once on 401.
        """
        url = f"{self.passport_server_url}{path}"
        response = requests.get(url, params=params, headers=self._headers())
        if response.status_code == 401:
            self.token = self._authenticate()
            response = requests.get(url, params=params, headers=self._headers())
        return response

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST to a Passport endpoint (studyId is added as query parameter), re-authenticating once on 401.
        """
        url = f"{self.passport_server_url}{path}?studyId={self.study_id}"
        headers = self._headers()
        response = requests.post(url, json=payload, headers=headers)
        response = self._refreshTokenAndRetry(response, headers, payload, url)
        if not response.ok:
            raise requests.HTTPError(f"{response.status_code} on POST {path}: {response.text}", response=response)
        return response.json()

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
        Obtain an access token for this component's Keycloak service account.

        :return token: The access token used as a bearer token for every Passport call.
        """
        token_url = f"{self.keycloak_server_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token"

        response = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        })
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
        objective_function = algorithm.objectiveFunction
        if objective_function is not None and not isinstance(objective_function, str):
            # e.g. a Keras loss given as a function or Loss object
            objective_function = getattr(objective_function, "__name__", None) or str(objective_function)
        payload = {"name": algorithm.name, "objectiveFunction": objective_function,
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
                                                  datasetTransformationId=response_learning_dataset.get(
                                                      'datasetTransformationId'),
                                                  learningDatasetId=response_learning_dataset.get('learningDatasetId'))

        return_dataset_transformation = DatasetTransformation(title=response_dataset_transformation.get('title'),
                                                              description=response_dataset_transformation.get(
                                                                  'description'),
                                                              datasetTransformationId=response_dataset_transformation.get(
                                                                  'datasetTransformationId'))
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
                   "datasetTransformationId": dataset_transformation_step.datasetTransformationId,
                   "stepOrder": dataset_transformation_step.stepOrder,
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
        return DatasetTransformationStep(datasetTransformationId=response_json.get('datasetTransformationId'),
                                         stepOrder=response_json.get('stepOrder'),
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
            "ownerOrganizationId": model.ownerOrganizationId,
            "previousModelId": model.previousModelId,
            "retrainingReason": model.retrainingReason,
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
            ownerOrganizationId=response_json.get('ownerOrganizationId'),
            previousModelId=response_json.get('previousModelId'),
            retrainingReason=response_json.get('retrainingReason')
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

    def submit_model_evaluation(self, model_evaluation: ModelEvaluation) -> ModelEvaluation:
        """
        Submit an evaluation run to the AI4HF Passport Server. Measures belong to a run rather than
        directly to the model, so a run must exist before any measure can be submitted.

        :param model_evaluation: ModelEvaluation object to be sent.

        :return ModelEvaluation: Created ModelEvaluation object from the server.
        """
        url = f"{self.passport_server_url}/model-evaluation?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "modelId": model_evaluation.modelId,
            "organizationId": model_evaluation.organizationId,
            "trigger": model_evaluation.trigger,
            "aggregationMethod": model_evaluation.aggregationMethod,
            "executedAt": model_evaluation.executedAt,
            "executedBy": model_evaluation.executedBy,
            "description": model_evaluation.description
        }

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return ModelEvaluation(
            modelEvaluationId=response_json.get('modelEvaluationId'),
            modelId=response_json.get('modelId'),
            organizationId=response_json.get('organizationId'),
            trigger=response_json.get('trigger'),
            aggregationMethod=response_json.get('aggregationMethod'),
            executedAt=response_json.get('executedAt'),
            executedBy=response_json.get('executedBy'),
            description=response_json.get('description'))

    def submit_model_evaluation_dataset(self, model_evaluation_dataset: ModelEvaluationDataset):
        """
        Link a LearningDataset the evaluation run was computed over.

        :param model_evaluation_dataset: ModelEvaluationDataset object to be sent.

        :return response: The response of the server for creating the link.
        """
        url = f"{self.passport_server_url}/model-evaluation-dataset?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "modelEvaluationId": model_evaluation_dataset.modelEvaluationId,
            "learningDatasetId": model_evaluation_dataset.learningDatasetId,
            "weight": model_evaluation_dataset.weight,
            "description": model_evaluation_dataset.description
        }

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        return response.json()

    def submit_evaluation_measure(self, evaluation_measure: EvaluationMeasure):
        """
        Submit evaluation measure to the AI4HF Passport Server.

        :param evaluation_measure: Dictionary object that holds evaluation measure related fields.

        :return response: The response of the server for creating the evaluation measure.
        """

        url = f"{self.passport_server_url}/evaluation-measure?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {
            "modelEvaluationId": evaluation_measure.modelEvaluationId,
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

    def submit_learning_process_dataset(self, learning_process_id: str, learning_dataset_id: str,
                                        description: Optional[str] = None) -> Dict[str, Any]:
        """
        Link a LearningDataset to the LearningProcess that trained on it. Together with ModelEvaluationDataset
        this is what the model-scoped passport uses to find the datasets behind a model.

        :return response: The response of the server for creating the link.
        """
        return self._post("/learning-process-dataset", {"learningProcessId": learning_process_id,
                                                        "learningDatasetId": learning_dataset_id,
                                                        "description": description})

    def resolve_executed_by(self) -> str:
        """
        The value recorded as executedBy of the evaluation runs this client creates: the id of the
        SoftwareAgent registered for this Keycloak client, as in the Passport's seed data. Falls back to the
        client id when the registry cannot be read (it needs a realm role, which a service account that is
        only a study-group member does not have).
        """
        try:
            response = self._get("/software-agent")
            if response.ok:
                for agent in response.json():
                    if agent.get("keycloakClientId") == self.client_id:
                        return agent.get("softwareAgentId")
        except requests.RequestException:
            pass
        return self.client_id

    def get_or_create_parameter(self, name: str, data_type: str, description: Optional[str]) -> Parameter:
        """
        Parameters are study-level definitions (a name and a data type) shared by learning processes,
        stages and models. Reuse the study's parameter of the same name instead of adding a duplicate
        definition on every run; create it when it does not exist yet.
        """
        if self._parameter_cache is None:
            self._parameter_cache = {}
            response = self._get("/parameter", {"studyId": self.study_id})
            if response.ok:
                for p in response.json():
                    self._parameter_cache.setdefault(p.get("name"), Parameter(
                        p.get("parameterId"), p.get("studyId"), p.get("name"), p.get("dataType"),
                        p.get("description")))
        existing = self._parameter_cache.get(name)
        if existing is not None:
            return existing
        created = self.submit_parameter(Parameter("", self.study_id, name, data_type, description))
        self._parameter_cache[name] = created
        return created

    def submit_evaluation_run(self, model_id: str, model_evaluation: ModelEvaluation,
                              evaluation_measures: list[EvaluationMeasure],
                              learning_datasets: Optional[list[LearningDataset]] = None) -> ModelEvaluation:
        """
        Record one evaluation run of a model: the run, its measures, and the learning datasets it was computed
        over (with their weight). submit_results_to_ai4hf_passport records the model's main run with this; call
        it directly for further runs, e.g. one per federated site with organizationId set to that site.

        :param model_id: The model the run evaluated (the "modelId" returned by submit_results_to_ai4hf_passport).
        :param model_evaluation: The run. executedBy defaults to this client's SoftwareAgent.
        :param evaluation_measures: The metrics of the run.
        :param learning_datasets: Learning datasets (with learningDatasetId) the run was computed over.

        :return ModelEvaluation: The created run.
        """
        model_evaluation = copy.deepcopy(model_evaluation)
        model_evaluation.modelId = model_id
        if model_evaluation.executedBy is None:
            model_evaluation.executedBy = self.resolve_executed_by()
        created_model_evaluation = self.submit_model_evaluation(model_evaluation)
        print(f'Model evaluation created: {created_model_evaluation}')
        for evaluation_measure in copy.deepcopy(evaluation_measures or []):
            evaluation_measure.modelEvaluationId = created_model_evaluation.modelEvaluationId
            evaluation_measure_response = self.submit_evaluation_measure(evaluation_measure)
            print(f'Evaluation measure created: {evaluation_measure_response}')
        for dataset in learning_datasets or []:
            self.submit_model_evaluation_dataset(ModelEvaluationDataset(
                modelEvaluationId=created_model_evaluation.modelEvaluationId,
                learningDatasetId=dataset.learningDatasetId,
                weight=dataset.weight,
                description=dataset.description))
            print(f'Model Evaluation Dataset linked: {dataset.learningDatasetId}')
        return created_model_evaluation

    def submit_results_to_ai4hf_passport(self,
                                         model: Any,
                                         learning_stages: Optional[list[LearningStage]] = None,
                                         model_evaluation: Optional[ModelEvaluation] = None,
                                         evaluation_measures: Optional[list[EvaluationMeasure]] = None,
                                         model_info: Optional[Model] = None,
                                         learning_dataset: Union[LearningDataset, list[LearningDataset], None] = None,
                                         dataset_transformation: Optional[DatasetTransformation] = None,
                                         dataset_transformation_steps: Optional[list[DatasetTransformationStep]] = None,
                                         model_figures: Optional[list[ModelFigure]] = None,
                                         learning_process_parameters: Optional[list[LearningProcessParameter]] = None,
                                         learning_stage_parameters: Optional[list[LearningStageParameter]] = None
                                         ) -> Dict[str, Any]:
        """
        Submit results of the ML model to the AI4HF Passport Server: everything a model-scoped passport is
        generated from. Pass the arguments by keyword; only model, model_info, model_evaluation,
        evaluation_measures and learning_dataset are required. The given objects are not modified, so the same
        lists can be reused across calls.

        :param model: The trained model object (scikit-learn estimator, Keras model or torch.nn.Module).
        :param model_info: Descriptive fields of the model (name, version, license, intended use, ...).
        :param model_evaluation: The evaluation run the measures were produced in. For a federated
            aggregate leave organizationId empty; executedBy defaults to this client's SoftwareAgent.
        :param evaluation_measures: The metrics of that run.
        :param learning_dataset: The learning dataset the model was trained and evaluated on, or a list with one
            per participating site for a federated model. Each is linked to the learning process (trained on)
            and to the evaluation run (evaluated on, with its weight).
        :param learning_stages: Optional training/validation/test stages with their dataset percentages.
        :param dataset_transformation: Optional title/description of the transformation applied to the datasets.
            One transformation is created per new learning dataset (the Passport creates both together).
        :param dataset_transformation_steps: Optional ordered transformation steps; stepOrder follows the list.
        :param model_figures: Optional figures (see ModelFigure.from_matplotlib).
        :param learning_process_parameters: Optional parameters of the learning process; these appear in the
            passport.
        :param learning_stage_parameters: Optional parameters of individual stages (stage=LearningStageType.X);
            these appear in the passport.

        :return dict: Ids of the created records (learningProcessId, modelId, modelEvaluationId,
            learningDatasetIds) and the created LearningDataset objects (learningDatasets, in the given order).
        """
        missing = [name for name, value in (("model_info", model_info), ("model_evaluation", model_evaluation),
                                            ("evaluation_measures", evaluation_measures),
                                            ("learning_dataset", learning_dataset)) if value is None]
        if missing:
            raise ValueError(f"submit_results_to_ai4hf_passport needs: {', '.join(missing)}")
        if dataset_transformation is None:
            dataset_transformation = DatasetTransformation(title="Learning dataset preparation",
                                                           description="Preparation of the dataset for training.")

        learning_stages = copy.deepcopy(learning_stages or [])
        model_evaluation = copy.deepcopy(model_evaluation)
        evaluation_measures = copy.deepcopy(evaluation_measures)
        model_info = copy.deepcopy(model_info)
        learning_datasets = copy.deepcopy(learning_dataset if isinstance(learning_dataset, list) else [learning_dataset])
        dataset_transformation_steps = copy.deepcopy(dataset_transformation_steps or [])
        model_figures = copy.deepcopy(model_figures or [])
        learning_process_parameters = copy.deepcopy(learning_process_parameters or [])
        learning_stage_parameters = copy.deepcopy(learning_stage_parameters or [])

        print('Sending given informations into AI4HF Passport server....')

        # 1. Algorithm -> Implementation -> LearningProcess
        learning_process = self.extract_and_submit_learning_process(model)
        print(f'Learning process created: {learning_process}')
        for learning_process_parameter in learning_process_parameters:
            learning_process_parameter.learningProcessId = learning_process.learningProcessId

        # 2. Learning stages, and the stage parameters matched to them by stage type
        stage_ids: Dict[str, str] = {}
        for learning_stage in learning_stages:
            learning_stage.learningProcessId = learning_process.learningProcessId
            learning_stage_response = self.submit_learning_stage(learning_stage)
            stage_type = learning_stage.learningStageType
            for key in (str(stage_type), stage_type.value, stage_type.name):
                stage_ids[key] = learning_stage_response.get("learningStageId")
            print(f'Learning stage created: {learning_stage_response}')
        matched_learning_stage_parameters = []
        for learning_stage_parameter in learning_stage_parameters:
            key = learning_stage_parameter.learningStageId
            key = str(key) if isinstance(key, LearningStageType) else key
            if key in stage_ids:
                learning_stage_parameter.learningStageId = stage_ids[key]
                matched_learning_stage_parameters.append(learning_stage_parameter)
            else:
                warnings.warn(f"Learning stage parameter '{learning_stage_parameter.name}' refers to stage "
                              f"'{key}', which is not among the submitted learning stages; it is skipped.")

        # 3. Model. createdBy/lastUpdatedBy reference a Personnel record, and a service account is not a
        # person: machine authorship is recorded by the Passport itself, which resolves the token to a
        # SoftwareAgent and writes it into the audit log book.
        model_info.learningProcessId = learning_process.learningProcessId
        model_info.studyId = self.study_id
        model_info.experimentId = self.experiment_id
        model_info.ownerOrganizationId = self.organization_id
        model_info.createdBy = None
        model_info.lastUpdatedBy = None
        created_model: Model = self.extract_and_submit_model(model, model_info)
        print(f'Model created: {created_model}')

        # 4. Learning datasets (+ transformation and its ordered steps), linked to the learning process
        for position, step in enumerate(dataset_transformation_steps, start=1):
            if step.stepOrder is None:
                step.stepOrder = position
        created_learning_datasets: list[LearningDataset] = []
        for dataset in learning_datasets:
            if dataset.learningDatasetId:
                print(f'Learning Dataset reused: {dataset}')
                created_learning_datasets.append(dataset)
            else:
                response_learning_dataset, response_dataset_transformation = self.submit_learning_dataset_and_transformation(
                    learning_dataset=dataset, dataset_transformation=dataset_transformation)
                response_learning_dataset.weight = dataset.weight
                print(f'Learning Dataset created: {response_learning_dataset}')
                print(f'Dataset Transformation created: {response_dataset_transformation}')
                for step in dataset_transformation_steps:
                    step_to_submit = copy.copy(step)
                    step_to_submit.datasetTransformationId = response_dataset_transformation.datasetTransformationId
                    step_to_submit.createdBy = None
                    step_to_submit.lastUpdatedBy = None
                    response_step = self.submit_dataset_transformation_step(step_to_submit)
                    print(f'Dataset Transformation Step created: {response_step}')
                created_learning_datasets.append(response_learning_dataset)
            self.submit_learning_process_dataset(learning_process.learningProcessId,
                                                 created_learning_datasets[-1].learningDatasetId,
                                                 created_learning_datasets[-1].description)
            print(f'Learning Process Dataset linked: {created_learning_datasets[-1].learningDatasetId}')

        # 5. The evaluation run, its measures, and the datasets it was computed over
        created_model_evaluation = self.submit_evaluation_run(created_model.modelId, model_evaluation,
                                                              evaluation_measures, created_learning_datasets)

        # 6. Parameters
        self.extract_and_submit_parameters(model, created_model.modelId, learning_process_parameters,
                                           matched_learning_stage_parameters)

        # 7. Figures
        for model_figure in model_figures:
            model_figure.modelId = created_model.modelId
            model_figure_response = self.submit_model_figure(model_figure)
            print(f'Model Figure created: {model_figure_response}')

        print('Given informations are sent into AI4HF Passport server!')
        return {"learningProcessId": learning_process.learningProcessId,
                "modelId": created_model.modelId,
                "modelEvaluationId": created_model_evaluation.modelEvaluationId,
                "learningDatasetIds": [d.learningDatasetId for d in created_learning_datasets],
                "learningDatasets": created_learning_datasets}

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

    def submit_learning_stage_parameter(self, learning_stage_parameter: LearningStageParameter) -> LearningStageParameter:
        """
        Submit Learning Stage Parameter to the AI4HF Passport Server.

        :param learning_stage_parameter: Learning Stage Parameter object to be sent.

        :return LearningStageParameter: Created Learning Stage Parameter object from the server.
        """
        url = f"{self.passport_server_url}/learning-stage-parameter?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"learningStageId": learning_stage_parameter.learningStageId, "parameterId": learning_stage_parameter.parameterId, "type": learning_stage_parameter.type,
                   "value": learning_stage_parameter.value}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return LearningStageParameter(learning_stage_parameter.name,response_json.get("id").get('learningStageId'), response_json.get("id").get('parameterId'),
                         response_json.get('type'), response_json.get('value'), learning_stage_parameter.description)

    def submit_learning_process_parameter(self, learning_process_parameter: LearningProcessParameter) -> LearningProcessParameter:
        """
        Submit Learning Process Parameter to the AI4HF Passport Server.

        :param learning_process_parameter: Learning Process Parameter object to be sent.

        :return LearningProcessParameter: Created Learning Process Parameter object from the server.
        """
        url = f"{self.passport_server_url}/learning-process-parameter?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"learningProcessId": learning_process_parameter.learningProcessId, "parameterId": learning_process_parameter.parameterId, "type": learning_process_parameter.type,
                   "value": learning_process_parameter.value}
        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return LearningProcessParameter(learning_process_parameter.name, response_json.get("id").get('learningProcessId'), response_json.get("id").get('parameterId'),
                                      response_json.get('type'), response_json.get('value'), learning_process_parameter.description)

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
                                      model: Any, model_id: str,
                                      learning_process_parameters: list[LearningProcessParameter],
                                      learning_stage_parameters: list[LearningStageParameter]):
        """
        Extract and submit parameters into the server. Every parameter is linked to a study-level Parameter
        definition of the same name (reused when it exists). The link's type holds the Passport data type
        (int, decimal, boolean, string), as in the Passport's own data.

        :param model: Model class that specific to implemented library.
        :param model_id: The ID of the model.
        :param learning_process_parameters: List of Learning Process Parameter objects (learningProcessId set).
        :param learning_stage_parameters: List of Learning Stage Parameter objects (learningStageId set).
        """
        extracted_parameters = self.extract_parameters(model) or []
        if not self.include_learned_parameters:
            # "parameter" marks learned values (weights, coefficients, ...) in the framework extractors
            extracted_parameters = [p for p in extracted_parameters if p.get('type') != 'parameter']
        for extracted_parameter_dict in extracted_parameters:
            data_type = normalize_data_type(extracted_parameter_dict.get('dataType'))
            parameter = self.get_or_create_parameter(extracted_parameter_dict.get('name'), data_type,
                                                     extracted_parameter_dict.get('description'))
            model_parameter = ModelParameter(model_id, parameter.parameterId, data_type,
                                             extracted_parameter_dict.get('value', None))
            created_model_parameter = self.submit_model_parameter(model_parameter)
            print(f'Model Parameter created: {extracted_parameter_dict.get("name")} = {created_model_parameter.value}')

        for learning_process_parameter in learning_process_parameters:
            parameter = self.get_or_create_parameter(learning_process_parameter.name, learning_process_parameter.type,
                                                     learning_process_parameter.description)
            learning_process_parameter.parameterId = parameter.parameterId
            created_learning_process_parameter = self.submit_learning_process_parameter(learning_process_parameter)
            print(f'Learning Process Parameter created: {created_learning_process_parameter}')

        for learning_stage_parameter in learning_stage_parameters:
            parameter = self.get_or_create_parameter(learning_stage_parameter.name, learning_stage_parameter.type,
                                                     learning_stage_parameter.description)
            learning_stage_parameter.parameterId = parameter.parameterId
            created_learning_stage_parameter = self.submit_learning_stage_parameter(learning_stage_parameter)
            print(f'Learning Stage Parameter created: {created_learning_stage_parameter}')

    def submit_model_figure(self, model_figure: ModelFigure) -> ModelFigure:
        """
        Submit ModelFigure to the AI4HF Passport Server.

        :param model_figure: ModelFigure object to be sent.

        :return ModelFigure: Created ModelFigure object from the server.
        """
        url = f"{self.passport_server_url}/model-figure?studyId={self.study_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"modelId": model_figure.modelId, "title": model_figure.title,
                   "description": model_figure.description, "imageBase64": model_figure.imageBase64}

        response = requests.post(url, json=payload, headers=headers)

        # If token is expired, retry
        response = self._refreshTokenAndRetry(response, headers, payload, url)

        response.raise_for_status()
        response_json: dict = response.json()
        return ModelFigure(response_json.get('imageBase64'),
                           response_json.get('figureId'),
                           response_json.get('modelId'),
                           response_json.get('title'),
                           response_json.get('description'))