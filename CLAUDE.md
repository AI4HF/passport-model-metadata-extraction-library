# ML Metadata Extraction Library (`passport-model-metadata-extraction-library`)

A **Python client library** for the AI4HF Passport Server: after a training pipeline finishes, it extracts
metadata from the trained model object (Scikit-learn, TensorFlow/Keras, or PyTorch) and pushes the full
training provenance — learning process, stages, parameters, model, evaluation measures, learning dataset,
transformations, figures — to the Passport REST API in one call.

Remote: https://github.com/AI4HF/passport-model-metadata-extraction-library · plain Python sources —
**not packaged** (no `setup.py`/`pyproject.toml`, no PyPI); examples import via `sys.path` from `lib/`.
Deps (install manually): `pandas scikit-learn torch tensorflow requests pyjwt`.

Workspace context: this is the seed of the **"Passport Python client library"** through which FL Central
is supposed to register global models in the federated redesign
([`../design/use-case-flow.md`](../design/use-case-flow.md), integration #3 / open question Q11). Today it
targets single-site training against the `/user/connector/*` M2M flow of `../passport`.

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
examples/
  main_sklearn.py / main_keras.py / main_torch.py       # end-to-end usage per framework
  main_monitoring_platform_simulation.py                # generates models/measures to feed the
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
- `extract_parameters(model)` → list of parameter dicts (`type`: `"hyperparameter"` from `get_params()`
  and `"parameter"` for learned values — coefficients, feature importances, layer weights, support
  vectors, cluster centers…) — beware: for large models this means **one Passport POST per weight array**.

## `submit_results_to_ai4hf_passport(...)` — the orchestrator (write order matters)

1. Algorithm → Implementation → LearningProcess (from `extract_learning_process`)
2. LearningStages (caller-provided; `LearningStage` maps the `LearningStageType` enum to fixed
   names — "Model Training/Testing/Validation" — which are then **string-matched** to attach the caller's
   `LearningStageParameter`s, whose `learningStageId` must initially hold `str(LearningStageType.X)`)
3. Model (via `extract_model`; `ownerOrganizationId` = organization id, `createdBy` = unverified-decoded
   `sub` of the token; `previousModelId`/`retrainingReason` carry the retraining lineage)
4. ModelEvaluation — the run the measures were produced in. Measures no longer attach to the model
   directly, so the caller passes a `ModelEvaluation` and the run is created before any measure.
   `organizationId` stays unset for federated training aggregates.
5. EvaluationMeasures (caller-provided, from the `EvaluationMeasureType` enum — `F1_SCORE` serializes as
   `"f1"` for Monitoring Platform compatibility) — each stamped with the created run's id
6. Parameters: extracted model parameters (as ModelParameter) + caller's learning-process and
   learning-stage parameters (each first created as a study-level Parameter, then linked)
7. LearningDataset + DatasetTransformation (single combined endpoint) + TransformationSteps, then a
   ModelEvaluationDataset linking that learning dataset to the run
8. ModelFigures (base64 PNGs, with `title`/`description`)

Auth mirrors the connectors: `POST /user/connector/login` with the raw offline Keycloak token
(`connector_secret`) as body; one re-auth + retry on 401 (`_refreshTokenAndRetry`, hardwired to POST —
fine, every call here is a POST). The referenced `study_id`, `experiment_id`, `organization_id`, and the
`datasetId` behind the LearningDataset must already exist in the Passport.

## Gotchas

- **Not idempotent** — every invocation creates a fresh Algorithm/Implementation/LearningProcess/Model
  chain; nothing is looked up or reused.
- `examples/main_monitoring_platform_simulation.py` builds 20 models' worth of drifting measures in a loop
  but calls `submit_results_to_ai4hf_passport` once (last iteration's values) — run it repeatedly / adapt
  the loop to actually populate rounds on the monitoring dashboard.
- The README's `submit_results_to_ai4hf_passport` snippet shows the old 4-argument signature; the real
  method now requires 10 (learning dataset, transformation(+steps), figures, process/stage parameters).
- Demo `CONNECTOR_SECRET` tokens are committed in the README and examples (same debt pattern as the
  other repos).

## Conventions

Gitmoji commits, kebab-case branches off `main`, PRs. No automated tests — verify by running an example
end-to-end against a deployed Passport (`sh passport/docker/deployment/run.sh` + proxy) and checking the
created chain in the Passport UI.
