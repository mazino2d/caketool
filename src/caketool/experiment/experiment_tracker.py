import pickle
from typing import *
from google.cloud import aiplatform, storage


class ExperimentTracker:
    def __init__(
        self,
        project: str,
        location: str,
        experiment_name: str,
        experiment_run_name: str,
        bucket_name: str,
        type: Literal['develop', 'deploy'] = "develop",
        experiment_description: str = None,
        experiment_tensorboard: bool = False,
    ) -> None:
        self.project = project
        self.location = location
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_tensorboard = experiment_tensorboard
        self.experiment_run_name = experiment_run_name
        self.bucket_name = bucket_name
        self.type = type
        self.gs_client = storage.Client(project=self.project)
        self.gc_bucket = storage.Bucket(self.gs_client, self.bucket_name)
        aiplatform.init(
            project=self.project,
            location=self.location,
            experiment=self.experiment_name,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.bucket_name,
        )
        if self.type == "develop":
            self.experiment_run = aiplatform.start_run(self.experiment_run_name)
            self.execution = aiplatform.start_execution(
                display_name="Experiment Tracking",
                schema_title="system.ContainerExecution"
            )

    def log_params(self, params: Dict[str, Union[float, int, str]]):
        self.experiment_run.log_params(params)

    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        self.experiment_run.log_metrics(metrics)

    def log_file(self, filename: str, artifact_id: str):
        blob = self._create_blob(artifact_id)
        blob.upload_from_filename(filename)

    def log_pickle(self, model: object, artifact_id: str):
        pickle_out = pickle.dumps(model)
        blob = self._get_blob(artifact_id)
        self._add_artifact(artifact_id)
        blob.upload_from_string(pickle_out)

    def load_pickle(self, artifact_id: str) -> object:
        blob = self._get_blob(artifact_id)
        pickle_in = blob.download_as_string()
        return pickle.loads(pickle_in)

    def _get_blob(self, artifact_id: str) -> storage.Blob:
        blob_name = f"{self.experiment_name}-{self.experiment_run_name}-{artifact_id}"
        blob = self.gc_bucket.blob(blob_name)
        return blob

    def _add_artifact(self, artifact_id) -> storage.Blob:
        blob = self._get_blob(artifact_id)
        uri = blob.path_helper(self.bucket_name, blob.name)
        if blob.exists():
            raise f"{uri} existed! (Can not overwrite)"
        artifact = aiplatform.Artifact.create(
            uri=uri, schema_title="system.Artifact"
        )
        self.experiment_run._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[artifact.resource_name]
        )
        return blob

    def __enter__(self):
        if self.type == "develop":
            self.execution.__enter__()
            self.experiment_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.type == "develop":
            self.execution.__exit__(exc_type, exc_value, exc_traceback)
            self.experiment_run.__exit__(exc_type, exc_value, exc_traceback)
