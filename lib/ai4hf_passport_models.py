import json
from enum import Enum
from typing import Optional


class Algorithm:
    def __init__(self, name: str, objectiveFunction: str, type: str, subType: str, algorithmId: str):
        """
        Initialize the Algorithm object from arguments.
        """
        self.name = name
        self.objectiveFunction = objectiveFunction
        self.type = type
        self.subType = subType
        self.algorithmId = algorithmId

    def __str__(self):
        return json.dumps({"algorithmId": self.algorithmId, "name": self.name, "objectiveFunction": self.objectiveFunction, "type": self.type,
                          "subType": self.subType})

class Implementation:
    def __init__(self, name: str, software: str, description: str, algorithmId: str, implementationId: str):
        """
        Initialize the Implementation object from arguments.
        """
        self.name = name
        self.software = software
        self.description = description
        self.algorithmId = algorithmId
        self.implementationId = implementationId

    def __str__(self):
        return json.dumps({"name": self.name, "software": self.software, "description": self.description, "algorithmId": self.algorithmId,
                          "implementationId": self.implementationId})

class LearningProcess:
    def __init__(self, learningProcessId: str, studyId: str, implementationId: str, description: str):
        """
        Initialize the LearningProcess object from arguments.
        """
        self.learningProcessId = learningProcessId
        self.studyId = studyId
        self.implementationId = implementationId
        self.description = description

    def __str__(self):
        return json.dumps({"learningProcessId": self.learningProcessId, "studyId": self.studyId, "implementationId": self.implementationId, "description": self.description})

class LearningProcessParameter:
    def __init__(self, name: str, learningProcessId: str, parameterId: str, type: str, value: str, description: str):
        """
        Initialize the LearningProcess object from arguments.
        """
        self.name = name
        self.learningProcessId = learningProcessId
        self.parameterId = parameterId
        self.type = type
        self.value = value
        self.description = description

    def __str__(self):
        return json.dumps({"name": self.name, "learningProcessId": self.learningProcessId, "parameterId": self.parameterId, "type": self.type, "value": self.value, "description": self.description})

class LearningStageType(Enum):
    """
        Type of learning stages. It contains three type: TEST, VALIDATION, TRAINING
    """
    TEST = "test"
    VALIDATION = "validation"
    TRAINING = "training"

class LearningStage:
    def __init__(self, learningStageType: LearningStageType, datasetPercentage: int):
        """
        Initialize the LearningStage object from learning stage type and dataset percentage.
        """
        if learningStageType == LearningStageType.TEST:
            self.learningStageName = "Model Testing"
            self.description = "Trains the model on the dataset"
        elif learningStageType == LearningStageType.TRAINING:
            self.learningStageName = "Model Training"
            self.description = "Training the model on the dataset"
        elif learningStageType == LearningStageType.VALIDATION:
            self.learningStageName = "Model Validation"
            self.description = "Validation dataset of the model"

        self.datasetPercentage = datasetPercentage
        self.learningStageId = None
        self.learningProcessId = None

    def __str__(self):
        return json.dumps({"learningStageId": self.learningStageId, "learningProcessId": self.learningProcessId, "learningStageName": self.learningStageName, "datasetPercentage": self.datasetPercentage, "description": self.description})

class LearningStageParameter:
    def __init__(self, name: str, learningStageId: str, parameterId: str, type: str, value: str, description: str):
        """
        Initialize the LearningProcess object from arguments.
        """
        self.name = name
        self.learningStageId = learningStageId
        self.parameterId = parameterId
        self.type = type
        self.value = value
        self.description = description

    def __str__(self):
        return json.dumps({"name": self.name, "learningStageId": self.learningStageId, "parameterId": self.parameterId, "type": self.type, "value": self.value, "description": self.description})
class Model:
    def __init__(self,
                 version: str = "",
                 tag: str = "",
                 productIdentifier: str = "",
                 trlLevel:str = "",
                 license: str = "",
                 primaryUse: str = "",
                 secondaryUse: str = "",
                 intendedUsers: str = "",
                 counterIndications: str = "",
                 ethicalConsiderations: str = "",
                 limitations: str = "",
                 fairnessConstraints: str = "",
                 createdBy: str = None,
                 lastUpdatedBy: str = None,
                 modelId: str = None,
                 learningProcessId: str = None,
                 studyId: str = None,
                 experimentId: str = None,
                 name: str = None,
                 ownerOrganizationId: str = None,
                 previousModelId: str = None,
                 retrainingReason: str = None,
                 modelType: str = None):
        """
        Initialize the Model object from arguments.
        """
        self.modelId = modelId
        self.learningProcessId = learningProcessId
        self.studyId = studyId
        self.experimentId = experimentId
        self.name = name
        self.version = version
        self.tag = tag
        self.modelType = modelType
        self.productIdentifier = productIdentifier
        self.ownerOrganizationId = ownerOrganizationId
        self.previousModelId = previousModelId
        self.retrainingReason = retrainingReason
        self.trlLevel = trlLevel
        self.license = license
        self.primaryUse = primaryUse
        self.secondaryUse = secondaryUse
        self.intendedUsers = intendedUsers
        self.counterIndications = counterIndications
        self.ethicalConsiderations = ethicalConsiderations
        self.limitations = limitations
        self.fairnessConstraints = fairnessConstraints
        self.createdBy = createdBy
        self.lastUpdatedBy = lastUpdatedBy

    def __str__(self):
        return json.dumps({"modelId": self.modelId,
                           "learningProcessId": self.learningProcessId,
                           "studyId": self.studyId,
                           "experimentId": self.experimentId,
                           "name": self.name,
                           "version": self.version,
                           "tag": self.tag,
                           "modelType": self.modelType,
                           "productIdentifier": self.productIdentifier,
                           "owner": self.owner,
                           "trlLevel": self.trlLevel,
                           "license": self.license,
                           "primaryUse": self.primaryUse,
                           "secondaryUse": self.secondaryUse,
                           "intendedUsers": self.intendedUsers,
                           "counterIndications": self.counterIndications,
                           "ethicalConsiderations": self.ethicalConsiderations,
                           "limitations": self.limitations,
                           "fairnessConstraints": self.fairnessConstraints,
                           "createdBy": self.createdBy,
                           "lastUpdatedBy": self.lastUpdatedBy})


class EvaluationMeasureType(Enum):
    """
        Type of evaluation metrics. It contains basic ones for regression and classification algorithms
    """
    TRUE_POSITIVES = "true_positives"
    TRUE_NEGATIVES = "true_negatives"
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"
    ACCURACY = "accuracy"
    AVERAGE_ACCURACY = "average_accuracy"
    ERROR_RATE = "error_rate"
    PRECISION = "precision"
    RECALL = "recall"
    SPECIFICITY = "specificity"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    F1_SCORE = "f1"
    ROC = "roc"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    RMSLE = "rmsle"
    R_SQUARED = "r_squared"
    ADJUSTED_R_SQUARED = "adjusted_r_squared",
    SENSITIVITY = "sensitivity"
    NPV = "npv"
    PPV = "ppv"

class ModelEvaluation:
    def __init__(self,
                 trigger: str = "training",
                 aggregationMethod: str = None,
                 executedAt: str = None,
                 executedBy: str = None,
                 description: str = None,
                 modelEvaluationId: Optional[str] = None,
                 modelId: Optional[str] = None,
                 organizationId: Optional[str] = None):
        """
        Initialize the ModelEvaluation object from arguments. One evaluation run of a model; the measures
        it produced hang off it, and the learning datasets it was computed over are linked through
        ModelEvaluationDataset. organizationId stays None for federated training aggregates.
        """
        self.modelEvaluationId = modelEvaluationId
        self.modelId = modelId
        self.organizationId = organizationId
        self.trigger = trigger
        self.aggregationMethod = aggregationMethod
        self.executedAt = executedAt
        self.executedBy = executedBy
        self.description = description

    def __str__(self):
        return json.dumps({"modelEvaluationId": self.modelEvaluationId, "modelId": self.modelId,
                           "organizationId": self.organizationId, "trigger": self.trigger,
                           "aggregationMethod": self.aggregationMethod, "executedAt": self.executedAt,
                           "executedBy": self.executedBy, "description": self.description})


class ModelEvaluationDataset:
    def __init__(self,
                 learningDatasetId: str,
                 weight: Optional[float] = None,
                 description: Optional[str] = None,
                 modelEvaluationId: Optional[str] = None):
        """
        Initialize the ModelEvaluationDataset object from arguments. A federated aggregate links one row
        per participating organization.
        """
        self.modelEvaluationId = modelEvaluationId
        self.learningDatasetId = learningDatasetId
        self.weight = weight
        self.description = description

    def __str__(self):
        return json.dumps({"modelEvaluationId": self.modelEvaluationId,
                           "learningDatasetId": self.learningDatasetId,
                           "weight": self.weight, "description": self.description})


class EvaluationMeasure:
    def __init__(self, evaluationMeasureType: EvaluationMeasureType, value: str):
        """
        Initialize the EvaluationMeasure object from evaluation measure type and the value.
        """
        self.name = evaluationMeasureType.value
        self.value = value
        self.dataType = "float"
        self.description = f"{evaluationMeasureType.value} of the model"
        self.modelEvaluationId = None
        self.measureId = None
        self.modelId = None

    def __str__(self):
        return json.dumps({"measureId": self.measureId, "modelId": self.modelId, "name": self.name, "value": self.value, "dataType": self.dataType, "description": self.description})

class Parameter:
    def __init__(self, parameterId: str, studyId: str, name: str, dataType: str, description: str):
        """
        Initialize the Parameter object from arguments.
        """
        self.parameterId = parameterId
        self.studyId = studyId
        self.name = name
        self.dataType = dataType
        self.description = description

    def __str__(self):
        return json.dumps({"parameterId": self.parameterId, "studyId": self.studyId, "name": self.name, "dataType": self.dataType, "description": self.description})

class ModelParameter:
    def __init__(self, modelId: str, parameterId: str, type: str, value: str):
        """
        Initialize the ModelParameter object from arguments.
        """
        self.modelId = modelId
        self.parameterId = parameterId
        self.type = type
        self.value = value

    def __str__(self):
        return json.dumps({"modelId": self.modelId, "parameterId": self.parameterId, "type": self.type, "value": self.value})

class LearningDataset:
    def __init__(self, datasetId: str, description: str, datasetTransformationId: Optional[str] = None, learningDatasetId: Optional[str] = None, studyId: Optional[str] = None):
        """
        Initialize the LearningDataset object from arguments.
        """
        self.learningDatasetId = learningDatasetId
        self.datasetId = datasetId
        self.studyId = studyId
        self.datasetTransformationId = datasetTransformationId
        self.description = description

    def __str__(self):
        return json.dumps({"learningDatasetId": self.learningDatasetId, "datasetId": self.datasetId, "studyId": self.studyId, "datasetTransformationId": self.datasetTransformationId, "description": self.description})

class DatasetTransformation:
    def __init__(self, title: str, description: str, datasetTransformationId: Optional[str] = None):
        """
        Initialize the DatasetTransformation object from arguments.
        """
        self.datasetTransformationId = datasetTransformationId
        self.title = title
        self.description = description

    def __str__(self):
        return json.dumps({"datasetTransformationId": self.datasetTransformationId, "title": self.title, "description": self.description})

class DatasetTransformationStep:
    def __init__(self, inputFeatures: str, outputFeatures: str, method: str,
                 explanation: str,createdAt: Optional[str] = None, lastUpdatedAt: Optional[str] = None,
                 datasetTransformationId: Optional[str] = None, stepId: Optional[str] = None, createdBy: Optional[str] = None,
                 lastUpdatedBy: Optional[str] = None):
        """
        Initialize the DatasetTransformationStep object from arguments.
        """
        self.stepId = stepId
        self.datasetTransformationId = datasetTransformationId
        self.inputFeatures = inputFeatures
        self.outputFeatures = outputFeatures
        self.method = method
        self.explanation = explanation
        self.createdAt = createdAt
        self.createdBy = createdBy
        self.lastUpdatedBy = lastUpdatedBy
        self.lastUpdatedAt = lastUpdatedAt

    def __str__(self):
        return json.dumps({"stepId": self.stepId, "datasetTransformationId": self.datasetTransformationId,
                        "inputFeatures": self.inputFeatures, "outputFeatures": self.outputFeatures,
                        "method": self.method, "explanation": self.explanation, "createdBy": self.createdBy,
                        "createdAt": self.createdAt, "lastUpdatedBy": self.lastUpdatedBy, "lastUpdatedAt": self.lastUpdatedAt})


class ModelFigure:
    def __init__(self, imageBase64: str, figureId: Optional[str] = None, modelId: Optional[str] = None,
                 title: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize the ModelFigure object from arguments.
        """
        self.figureId = figureId
        self.modelId = modelId
        self.title = title
        self.description = description
        self.imageBase64 = imageBase64

    def __str__(self):
        return json.dumps({"figureId": self.figureId, "modelId": self.modelId, "title": self.title,
                           "description": self.description, "imageBase64": self.imageBase64})