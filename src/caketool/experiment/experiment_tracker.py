import pickle
from typing import *
from google.cloud import aiplatform, storage


class ExperimentTracker:
    """
    A class to manage and track machine learning experiments using Google Cloud AI Platform.

    Parameters
    ----------
    project : str
        The GCP project ID.
    location : str
        The location for the AI Platform services.
    experiment_name : str
        The name of the experiment.
    experiment_run_name : str
        The name of the experiment run.
    bucket_name : str
        The name of the GCS bucket to store artifacts.
    mode : Literal['develop', 'deploy'], optional (default="develop")
        The mode of the experiment tracker. Can be "develop" or "deploy".
    experiment_description : str, optional (default=None)
        A description of the experiment.
    experiment_tensorboard : bool, optional (default=False)
        Whether to use TensorBoard for experiment tracking.
    """

    def __init__(
        self,
        project: str,
        location: str,
        experiment_name: str,
        experiment_run_name: str,
        bucket_name: str,
        mode: Literal['develop', 'deploy'] = "develop",
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
        self.mode = mode
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
        if self.mode == "develop":
            self.experiment_run = aiplatform.start_run(self.experiment_run_name)
            self.execution = aiplatform.start_execution(
                display_name="Experiment Tracking",
                schema_title="system.ContainerExecution"
            )

    def log_params(self, params: Dict[str, Union[float, int, str]]):
        """
        Log parameters to the experiment run.

        Parameters
        ----------
        params : Dict[str, Union[float, int, str]]
            A dictionary of parameters to log.
        """
        self.experiment_run.log_params(params)

    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        """
        Log metrics to the experiment run.

        Parameters
        ----------
        metrics : Dict[str, Union[float, int, str]]
            A dictionary of metrics to log.
        """
        self.experiment_run.log_metrics(metrics)

    def log_file(self, filename: str, artifact_id: str):
        """
        Log a file as an artifact to Google Cloud Storage.

        Parameters
        ----------
        filename : str
            The path to the file to log.
        artifact_id : str
            The unique identifier for the artifact.
        """
        blob = self._add_artifact(artifact_id)
        blob.upload_from_filename(filename)

    def log_pickle(self, var: object, artifact_id: str):
        """
        Log a pickled model as an artifact to Google Cloud Storage.

        Parameters
        ----------
        model : object
            The model object to pickle and log.
        artifact_id : str
            The unique identifier for the artifact.
        """
        pickle_out = pickle.dumps(var)
        blob = self._add_artifact(artifact_id)
        blob.upload_from_string(pickle_out)

    def load_pickle(self, artifact_id: str) -> object:
        """
        Load a pickled model from Google Cloud Storage.

        Parameters
        ----------
        artifact_id : str
            The unique identifier for the artifact.

        Returns
        -------
        object
            The unpickled model object.
        """
        blob = self._get_blob(artifact_id)
        pickle_in = blob.download_as_string()
        return pickle.loads(pickle_in)

    def _get_blob(self, artifact_id: str) -> storage.Blob:
        """
        Get a GCS blob for the specified artifact.

        Parameters
        ----------
        artifact_id : str
            The unique identifier for the artifact.

        Returns
        -------
        storage.Blob
            The GCS blob for the artifact.
        """
        blob_name = f"{self.experiment_name}-{self.experiment_run_name}-{artifact_id}"
        blob = self.gc_bucket.blob(blob_name)
        return blob

    def _add_artifact(self, artifact_id: str) -> storage.Blob:
        """
        Add an artifact to the experiment run in AI Platform.

        Parameters
        ----------
        artifact_id : str
            The unique identifier for the artifact.

        Returns
        -------
        storage.Blob
            The GCS blob for the artifact.
        """
        blob = self._get_blob(artifact_id)
        uri = blob.path_helper(self.bucket_name, blob.name)
        if blob.exists():
            raise ValueError(f"{uri} existed! (Cannot overwrite)")
        artifact = aiplatform.Artifact.create(
            uri=uri, schema_title="system.Artifact"
        )
        self.experiment_run._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[artifact.resource_name]
        )
        return blob

    def __enter__(self):
        """
        Enter the runtime context for the experiment tracker, starting the experiment run and execution.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.mode == "develop":
            self.execution.__enter__()
            self.experiment_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit the runtime context for the experiment tracker, ending the experiment run and execution.

        Parameters
        ----------
        exc_type : Type[BaseException]
            The exception type.

        exc_value : BaseException
            The exception value.

        exc_traceback : TracebackType
            The traceback object.
        """
        if self.mode == "develop":
            self.execution.__exit__(exc_type, exc_value, exc_traceback)
            self.experiment_run.__exit__(exc_type, exc_value, exc_traceback)
