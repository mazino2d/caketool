from typing import Dict, Union
from google.cloud import aiplatform, storage


class ExperimentTracker:
    def __init__(
        self,
        project: str,
        location: str,
        experiment_name: str,
        experiment_run_name: str,
        bucket_name: str,
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

        aiplatform.init(
            project=self.project,
            location=self.location,
            experiment=self.experiment_name,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.bucket_name,
        )

        self.experiment_run = aiplatform.start_run(self.experiment_run_name)
        self.execution = aiplatform.start_execution(
            display_name="Experiment Tracking",
            schema_title="system.ContainerExecution"
        )
        self.gs_client = storage.Client(project=self.project)
        self.gc_bucket = storage.Bucket(self.gs_client, self.bucket_name)

    def log_params(self, params: Dict[str, Union[float, int, str]]):
        aiplatform.log_params(params)

    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        aiplatform.log_metrics(metrics)

    def log_file(self, filename, artifact_id):
        blob_name = f"{self.experiment_name}-{self.experiment_run_name}-{artifact_id}"
        blob = self.gc_bucket.blob(blob_name)
        uri = "gs://" + storage.Blob.path_helper(self.bucket_name, blob_name)
        if blob.exists():
            raise f"{uri} existed! (Can not overwrite)"
        blob.upload_from_filename(filename)
        artifact = aiplatform.Artifact.create(
            uri=uri, schema_title="system.Artifact")
        self.experiment_run._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[artifact.resource_name])

    def __enter__(self):
        self.execution.__enter__()
        self.experiment_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.execution.__exit__(exc_type, exc_value, exc_traceback)
        self.experiment_run.__exit__(exc_type, exc_value, exc_traceback)
