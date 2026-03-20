import pickle
from abc import ABC, abstractmethod
from typing import Literal

from caketool.utils.lib_utils import require_dependencies


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.

    Provides a unified interface for logging parameters, metrics, and artifacts
    to different experiment tracking backends (Vertex AI, MLflow, etc.).

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    run_name : str
        The name of the experiment run.
    mode : Literal['develop', 'deploy'], optional (default="develop")
        The mode of the experiment tracker. "develop" enables logging,
        "deploy" disables logging for production use.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        mode: Literal["develop", "deploy"] = "develop",
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mode = mode

    @abstractmethod
    def log_params(self, params: dict[str, float | int | str]) -> None:
        """Log parameters to the experiment run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """Log metrics to the experiment run, optionally at a specific step."""
        pass

    @abstractmethod
    def log_file(self, filename: str, artifact_id: str) -> None:
        """Log a file as an artifact."""
        pass

    @abstractmethod
    def log_pickle(self, var: object, artifact_id: str) -> None:
        """Log a pickled object as an artifact."""
        pass

    @abstractmethod
    def load_pickle(self, artifact_id: str) -> object:
        """Load a pickled object from artifacts."""
        pass

    @abstractmethod
    def __enter__(self):
        """Enter the runtime context."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit the runtime context."""
        pass


class VertexAITracker(ExperimentTracker):
    """
    Experiment tracker using Google Cloud Vertex AI.

    Parameters
    ----------
    project : str
        The GCP project ID.
    location : str
        The location for the AI Platform services.
    experiment_name : str
        The name of the experiment.
    run_name : str
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

    @require_dependencies("google.cloud.aiplatform", "google.cloud.storage")
    def __init__(
        self,
        project: str,
        location: str,
        experiment_name: str,
        run_name: str,
        bucket_name: str,
        mode: Literal["develop", "deploy"] = "develop",
        experiment_description: str = None,
        experiment_tensorboard: bool = False,
    ) -> None:
        from google.cloud import aiplatform, storage

        super().__init__(experiment_name, run_name, mode)
        self.project = project
        self.location = location
        self.experiment_description = experiment_description
        self.experiment_tensorboard = experiment_tensorboard
        self.bucket_name = bucket_name
        self.gs_client = storage.Client(project=self.project)
        self.gc_bucket = storage.Bucket(self.gs_client, self.bucket_name)
        self._aiplatform = aiplatform
        self._storage = storage

        aiplatform.init(
            project=self.project,
            location=self.location,
            experiment=self.experiment_name,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.bucket_name,
        )
        if self.mode == "develop":
            self.experiment_run = aiplatform.start_run(self.run_name)
            self.execution = aiplatform.start_execution(
                display_name="Experiment Tracking", schema_title="system.ContainerExecution"
            )

    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Log parameters to the experiment run.

        Parameters
        ----------
        params : dict[str, float | int | str]
            A dictionary of parameters to log.
        """
        if self.mode == "develop":
            self.experiment_run.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the experiment run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            A dictionary of metrics to log.
        step : int, optional
            The step/iteration number for time-series metrics.
            If provided, uses log_time_series_metrics for tracking over time.
        """
        if self.mode == "develop":
            if step is not None:
                self.experiment_run.log_time_series_metrics(metrics, step=step)
            else:
                self.experiment_run.log_metrics(metrics)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Log a file as an artifact to Google Cloud Storage.

        Parameters
        ----------
        filename : str
            The path to the file to log.
        artifact_id : str
            The unique identifier for the artifact.
        """
        if self.mode == "develop":
            blob = self._add_artifact(artifact_id)
            blob.upload_from_filename(filename)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Log a pickled object as an artifact to Google Cloud Storage.

        Parameters
        ----------
        var : object
            The object to pickle and log.
        artifact_id : str
            The unique identifier for the artifact.
        """
        if self.mode == "develop":
            pickle_out = pickle.dumps(var)
            blob = self._add_artifact(artifact_id)
            blob.upload_from_string(pickle_out)

    def load_pickle(self, artifact_id: str) -> object:
        """
        Load a pickled object from Google Cloud Storage.

        Parameters
        ----------
        artifact_id : str
            The unique identifier for the artifact.

        Returns
        -------
        object
            The unpickled object.
        """
        blob = self._get_blob(artifact_id)
        pickle_in = blob.download_as_string()
        return pickle.loads(pickle_in)

    def _get_blob(self, artifact_id: str):
        """Get a GCS blob for the specified artifact."""
        blob_name = f"{self.experiment_name}-{self.run_name}-{artifact_id}"
        blob = self.gc_bucket.blob(blob_name)
        return blob

    def _add_artifact(self, artifact_id: str):
        """Add an artifact to the experiment run in AI Platform."""
        blob = self._get_blob(artifact_id)
        uri = blob.path_helper(self.bucket_name, blob.name)
        if blob.exists():
            raise ValueError(f"{uri} existed! (Cannot overwrite)")
        artifact = self._aiplatform.Artifact.create(uri=uri, schema_title="system.Artifact")
        self.experiment_run._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[artifact.resource_name]
        )
        return blob

    def __enter__(self):
        """Enter the runtime context for the experiment tracker."""
        if self.mode == "develop":
            self.execution.__enter__()
            self.experiment_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit the runtime context for the experiment tracker."""
        if self.mode == "develop":
            self.execution.__exit__(exc_type, exc_value, exc_traceback)
            self.experiment_run.__exit__(exc_type, exc_value, exc_traceback)


class MLflowTracker(ExperimentTracker):
    """
    Experiment tracker using MLflow.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    run_name : str
        The name of the experiment run.
    tracking_uri : str, optional (default=None)
        The URI of the MLflow tracking server.
        If None, uses the default local tracking.
    artifact_location : str, optional (default=None)
        The location to store artifacts. Can be local path or cloud storage URI.
    mode : Literal['develop', 'deploy'], optional (default="develop")
        The mode of the experiment tracker.
    tags : dict[str, str], optional (default=None)
        Tags to associate with the experiment run.
    """

    @require_dependencies("mlflow")
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str = None,
        artifact_location: str = None,
        mode: Literal["develop", "deploy"] = "develop",
        tags: dict[str, str] = None,
    ) -> None:
        import mlflow

        super().__init__(experiment_name, run_name, mode)
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.tags = tags or {}
        self._mlflow = mlflow
        self._run = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            self.experiment_id = experiment.experiment_id

    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Log parameters to the experiment run.

        Parameters
        ----------
        params : dict[str, float | int | str]
            A dictionary of parameters to log.
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the experiment run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            A dictionary of metrics to log.
        step : int, optional
            The step/iteration number for time-series metrics (e.g., epoch).
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_metrics(metrics, step=step)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Log a file as an artifact to MLflow.

        Parameters
        ----------
        filename : str
            The path to the file to log.
        artifact_id : str
            The artifact path/subdirectory in MLflow.
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_artifact(filename, artifact_path=artifact_id)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Log a pickled object as an artifact to MLflow.

        Parameters
        ----------
        var : object
            The object to pickle and log.
        artifact_id : str
            The unique identifier for the artifact.
        """
        if self.mode == "develop" and self._run:
            import os
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"{artifact_id}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(var, f)
                self._mlflow.log_artifact(filepath)

    def load_pickle(self, artifact_id: str) -> object:
        """
        Load a pickled object from MLflow artifacts.

        Parameters
        ----------
        artifact_id : str
            The unique identifier for the artifact.

        Returns
        -------
        object
            The unpickled object.
        """
        if self._run is None:
            raise RuntimeError("No active run. Use within context manager or start a run first.")

        artifact_uri = f"{self._run.info.artifact_uri}/{artifact_id}.pkl"
        local_path = self._mlflow.artifacts.download_artifacts(artifact_uri)
        with open(local_path, "rb") as f:
            return pickle.load(f)

    def __enter__(self):
        """Enter the runtime context for the experiment tracker."""
        if self.mode == "develop":
            self._run = self._mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                tags=self.tags,
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit the runtime context for the experiment tracker."""
        if self.mode == "develop" and self._run:
            self._mlflow.end_run()
            self._run = None


class WandbTracker(ExperimentTracker):
    """
    Experiment tracker using Weights & Biases (wandb).

    Parameters
    ----------
    experiment_name : str
        The name of the experiment (maps to wandb project).
    run_name : str
        The name of the experiment run.
    mode : Literal['develop', 'deploy'], optional (default="develop")
        The mode of the experiment tracker. "develop" enables logging,
        "deploy" disables logging.
    entity : str, optional (default=None)
        The wandb entity (username or team name).
    tags : list[str], optional (default=None)
        Tags to associate with the run.
    config : dict, optional (default=None)
        Initial config/hyperparameters to log.
    """

    @require_dependencies("wandb")
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        mode: Literal["develop", "deploy"] = "develop",
        entity: str = None,
        tags: list[str] = None,
        config: dict = None,
    ) -> None:
        import wandb

        super().__init__(experiment_name, run_name, mode)
        self.entity = entity
        self.tags = tags
        self.config = config or {}
        self._wandb = wandb
        self._run = None

    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Log parameters to the wandb run config.

        Parameters
        ----------
        params : dict[str, float | int | str]
            A dictionary of parameters to log.
        """
        if self.mode == "develop" and self._run:
            self._run.config.update(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the wandb run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            A dictionary of metrics to log.
        step : int, optional
            The step/iteration number.
        """
        if self.mode == "develop" and self._run:
            self._run.log(metrics, step=step)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Log a file as a wandb artifact.

        Parameters
        ----------
        filename : str
            The path to the file to log.
        artifact_id : str
            The name for the artifact.
        """
        if self.mode == "develop" and self._run:
            artifact = self._wandb.Artifact(artifact_id, type="file")
            artifact.add_file(filename)
            self._run.log_artifact(artifact)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Log a pickled object as a wandb artifact.

        Parameters
        ----------
        var : object
            The object to pickle and log.
        artifact_id : str
            The name for the artifact.
        """
        if self.mode == "develop" and self._run:
            import os
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"{artifact_id}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(var, f)
                artifact = self._wandb.Artifact(artifact_id, type="pickle")
                artifact.add_file(filepath)
                self._run.log_artifact(artifact)

    def load_pickle(self, artifact_id: str) -> object:
        """
        Load a pickled object from a wandb artifact.

        Parameters
        ----------
        artifact_id : str
            The name of the artifact to load.

        Returns
        -------
        object
            The unpickled object.
        """
        if self._run is None:
            raise RuntimeError("No active run. Use within context manager or start a run first.")

        artifact = self._run.use_artifact(f"{artifact_id}:latest")
        artifact_dir = artifact.download()
        filepath = f"{artifact_dir}/{artifact_id}.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __enter__(self):
        """Enter the runtime context for the experiment tracker."""
        if self.mode == "develop":
            self._run = self._wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                entity=self.entity,
                tags=self.tags,
                config=self.config,
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit the runtime context for the experiment tracker."""
        if self.mode == "develop" and self._run:
            self._run.finish()
            self._run = None


def create_tracker(
    backend: Literal["vertex_ai", "mlflow", "wandb"],
    experiment_name: str,
    run_name: str,
    mode: Literal["develop", "deploy"] = "develop",
    **kwargs,
) -> ExperimentTracker:
    """
    Factory function to create an experiment tracker.

    Parameters
    ----------
    backend : Literal["vertex_ai", "mlflow"]
        The backend to use for experiment tracking.
    experiment_name : str
        The name of the experiment.
    run_name : str
        The name of the experiment run.
    mode : Literal['develop', 'deploy'], optional (default="develop")
        The mode of the experiment tracker.
    **kwargs
        Additional arguments specific to the backend.

        For vertex_ai:
            - project: str - GCP project ID
            - location: str - GCP location
            - bucket_name: str - GCS bucket name
            - experiment_description: str - Experiment description
            - experiment_tensorboard: bool - Use TensorBoard

        For mlflow:
            - tracking_uri: str - MLflow tracking server URI
            - artifact_location: str - Artifact storage location
            - tags: dict[str, str] - Run tags

        For wandb:
            - entity: str - wandb entity (username or team)
            - tags: list[str] - Run tags
            - config: dict - Initial hyperparameters

    Returns
    -------
    ExperimentTracker
        An instance of the appropriate tracker class.

    Examples
    --------
    >>> # Vertex AI
    >>> tracker = create_tracker(
    ...     backend="vertex_ai",
    ...     experiment_name="my-exp",
    ...     run_name="run-001",
    ...     project="my-gcp-project",
    ...     location="us-central1",
    ...     bucket_name="my-bucket",
    ... )

    >>> # MLflow
    >>> tracker = create_tracker(
    ...     backend="mlflow",
    ...     experiment_name="my-exp",
    ...     run_name="run-001",
    ...     tracking_uri="http://localhost:5000",
    ... )

    >>> # Weights & Biases
    >>> tracker = create_tracker(
    ...     backend="wandb",
    ...     experiment_name="my-project",
    ...     run_name="run-001",
    ...     entity="my-team",
    ... )
    """
    if backend == "vertex_ai":
        return VertexAITracker(
            experiment_name=experiment_name,
            run_name=run_name,
            mode=mode,
            project=kwargs.get("project"),
            location=kwargs.get("location"),
            bucket_name=kwargs.get("bucket_name"),
            experiment_description=kwargs.get("experiment_description"),
            experiment_tensorboard=kwargs.get("experiment_tensorboard", False),
        )
    elif backend == "mlflow":
        return MLflowTracker(
            experiment_name=experiment_name,
            run_name=run_name,
            mode=mode,
            tracking_uri=kwargs.get("tracking_uri"),
            artifact_location=kwargs.get("artifact_location"),
            tags=kwargs.get("tags"),
        )
    elif backend == "wandb":
        return WandbTracker(
            experiment_name=experiment_name,
            run_name=run_name,
            mode=mode,
            entity=kwargs.get("entity"),
            tags=kwargs.get("tags"),
            config=kwargs.get("config"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'vertex_ai', 'mlflow', or 'wandb'.")
