# ML Metadata Extraction Library (`passport-model-metadata-extraction-library`)

A **Python client library** for the AI4HF Passport Server: after a training pipeline finishes, it extracts
metadata from the trained model object (Scikit-learn, TensorFlow/Keras, or PyTorch) and pushes the full
training provenance — learning process, stages, parameters, model, evaluation measures, learning dataset,
transformations, figures — to the Passport REST API in one call.

Remote: https://github.com/AI4HF/passport-model-metadata-extraction-library · plain Python sources —
**not packaged** (no `setup.py`/`pyproject.toml`, no PyPI); examples import via `sys.path` from `lib/`.
Deps (install manually): `pandas scikit-learn torch tensorflow requests matplotlib`.

Workspace context: this is the seed of the **"Passport Python client library"** through which FL Central
is supposed to register global models in the federated redesign
([`../design/use-case-flow.md`](../design/use-case-flow.md), integration #3 / open question Q11). Today it
handles single-site training and, via one learning dataset per site plus `submit_evaluation_run` for
per-site runs, a federated global model; it authenticates as the `ai4hf-fl-central` Keycloak service account.

## Layout

```
lib/
  ai4hf_passport_base.py      # BaseMetadataCollectionAPI — auth + one submit_* method per Passport entity
                              #   + the submit_results_to_ai4hf_passport(...) orchestrator
  ai4hf_passport_models.py    # plain-Python mirrors of Passport entities + LearningStageType /
                              #   EvaluationMeasureType enums
  ai4hf_passport_sklearn.py   # SKLearnMetadataCollectionAPI(BaseMetadataCollectionAPI)
  ai4hf_passport_keras.py     # KerasMetadataCollectionAPI
  ai4hf_passport_torch.py     # TorchMetadataCollectionAPI
examples/                                               # self-contained templates (each puts lib/ on sys.path)
  main_sklearn.py                                       # every optional argument, with real training
  main_keras.py / main_torch.py                         # required arguments only
  main_monitoring_platform_simulation.py                # N models with drifting measures for the
                                                        #   passport-monitoring-platform-connector demo
test-data/Social_Network_Ads.csv                        # toy dataset for the examples
```

## The extraction contract

`BaseMetadataCollectionAPI` handles everything Passport-side; each framework subclass overrides exactly
three hooks:

- `extract_learning_process(model)` → dict with `implementation{name, software, description, algorithm{…}}`
  (sklearn: estimator type/criterion; keras: layer/activation/loss/metrics info; torch: module structure)
- `extract_model(model, model_info)` → `Model` (auto-fills `modelType` etc.; the caller's `model_info`
  supplies the FactSheet-style fields: license, TRL, intended users, limitations, …)
- `extract_parameters(model)` → list of parameter dicts (`type`: `"hyperparameter"` / `"layer"` /
  `"layer parameter"` for configuration, `"parameter"` for learned values — coefficients, feature
  importances, layer weights…). The base class drops `"parameter"` entries unless the client was built with
  `include_learned_parameters=True` (one Passport POST pair per weight array, unreadable in a passport), maps
  `dataType` onto the Passport's `int`/`decimal`/`boolean`/`string` and writes that as the link's `type`.

## `submit_results_to_ai4hf_passport(...)` — the orchestrator (write order matters)

Pass arguments by keyword; only `model`, `model_info`, `model_evaluation`, `evaluation_measures` and
`learning_dataset` are required (ValueError otherwise; a generic DatasetTransformation is used when none is
given). Inputs are deep-copied (callers can reuse their lists across calls); returns
`{learningProcessId, modelId, modelEvaluationId, learningDatasetIds, learningDatasets}`.

Convenience in the models: `LearningProcessParameter(name, value=...)` / `LearningStageParameter(name, value=...,
stage=LearningStageType.X)` infer `type` from the value (`normalize_data_type`); `EvaluationMeasure` accepts
numbers (rounded to 4 decimals); `ModelFigure.from_matplotlib(fig, title, description)`.

1. Algorithm → Implementation → LearningProcess (from `extract_learning_process`)
2. LearningStages (caller-provided). `LearningStageParameter.learningStageId` must initially hold
   `str(LearningStageType.X)` (the enum, its value or name also work); it is matched on the stage's
   `learningStageType` and swapped for the created id — unmatched parameters are skipped with a warning
3. Model (via `extract_model`; `ownerOrganizationId` = organization id; `createdBy`/`lastUpdatedBy` stay
   null — they are FKs to personnel and a service account is not a person; the Passport attributes the
   write to the SoftwareAgent behind the token's `azp`. `previousModelId`/`retrainingReason` = lineage)
4. Learning datasets — one, or a list (one per federated site). Each without `learningDatasetId` is created
   with the combined LearningDataset+DatasetTransformation endpoint (the only create path, so N sites → N
   transformation copies) plus the TransformationSteps (`stepOrder` = list position unless set); one with
   `learningDatasetId` is reused as is. Each is linked with **LearningProcessDataset** (trained on)
5. `submit_evaluation_run(model_id, run, measures, learning_datasets)` (public — also used for per-site runs)
   — the run; `executedBy` defaults to `resolve_executed_by()` (the SoftwareAgent id via
   `GET /software-agent`, which needs a realm role the service account lacks → falls back to the client
   id). Then its EvaluationMeasures (`F1_SCORE` serializes as `"f1"` for Monitoring Platform
   compatibility) and one **ModelEvaluationDataset** per learning dataset, carrying `LearningDataset.weight`
6. Parameters: model parameters (ModelParameter) + learning-process and learning-stage parameters, each
   linked to a study-level Parameter **reused by name** (`get_or_create_parameter`, cached per client)
7. ModelFigures (base64 PNGs, with `title`/`description`)

The model-scoped passport (`PassportService.resolveModelScope` in `../passport`) finds a model's datasets
only through LearningProcessDataset and ModelEvaluationDataset — both links are needed. Its `parameters`
section holds learning-process and learning-stage parameters only; **ModelParameters are deliberately not
rendered** (too many — they would bloat the passport). Whatever must show in a passport goes in as a
learning-process or learning-stage parameter.

Auth: Keycloak `client_credentials` for the `ai4hf-fl-central` client (realm `AI4HF-Authorization`); its
service account is a member of `study-<id>/DATA_SCIENTIST`, which every endpoint used here accepts. One
re-auth + retry on 401 (`_refreshTokenAndRetry` for POSTs, `_get` for GETs). The referenced `study_id`,
`experiment_id`, `organization_id`, and the `datasetId`s behind the learning datasets must already exist;
a learning dataset is assigned to the study of its dataset, not to `study_id`.

## Gotchas

- **Not idempotent** — every invocation creates a fresh Algorithm/Implementation/LearningProcess/Model
  chain and new learning datasets; only study-level Parameter definitions are reused.
- `srdc/passport:latest` / `srdc/passport-web:latest` images from before the September 2026 redesign fail
  against the current schema (e.g. `data_transformation_id does not exist`) — `docker pull` them (the
  2026-09-25 builds work).
  The examples default to the proxy (`passport/docker/proxy/run.sh`): API at
  `http://localhost/ai4hf/passport/api`, UI at `http://localhost/ai4hf/passport`; without the proxy the
  backend is on 8082 (docker) or 8080 (IDE) — set `PASSPORT_SERVER_URL`.
- Generating a passport with "exclude empty fields" off fails with an FK error for models that have a
  run without `organizationId` (`PassportService` writes "N/A" into managed entities) — a backend bug;
  the UI default (on) works.
- The demo client secret of `ai4hf-fl-central` (from the Passport's realm-import) is committed in the examples.

## Conventions

Gitmoji commits, kebab-case branches off `main`, PRs. No automated tests — verify by running an example
end-to-end against a deployed Passport (`sh passport/docker/deployment/run.sh` + `sh passport/docker/proxy/run.sh`,
from the workspace root) and generating the model's passport
(as `quality_assurance_specialist`) to check every section is filled.
