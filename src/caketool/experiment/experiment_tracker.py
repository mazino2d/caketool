import pickle
from abc import ABC, abstractmethod
from typing import Literal

from caketool.utils.lib_utils import require_dependencies


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.

    Provides a unified interface for logging parameters, metrics, and artifacts
    to different experiment tracking backends (Vertex AI, MLflow, wandb).

    Use this class as a context manager (``with`` statement) to automatically
    open and close a run. Set ``mode="deploy"`` to disable all logging in
    production without changing any other code.

    API keys and credentials should be set via environment variables, either
    directly in the shell or through a ``.env`` file passed via ``dotenv_path``.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment. Groups multiple runs together.
    run_name : str
        Name of this specific run within the experiment.
    mode : {"develop", "deploy"}, optional
        ``"develop"`` enables logging (default). ``"deploy"`` silently skips
        all logging — useful for production inference.
    dotenv_path : str or None, optional
        Path to a ``.env`` file to load before initializing the tracker.
        Useful for storing API keys and credentials outside the codebase.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        mode: Literal["develop", "deploy"] = "develop",
        dotenv_path: str | None = None,
    ) -> None:
        if dotenv_path is not None:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mode = mode

    @abstractmethod
    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Log hyperparameters for this run.

        Call this once at the start of a run to record the configuration
        (e.g., learning rate, batch size, regularization strength).

        Parameters
        ----------
        params : dict[str, float | int | str]
            Hyperparameter names and their values.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log evaluation metrics for this run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            Metric names and their values (e.g., ``{"accuracy": 0.95, "loss": 0.1}``).
        step : int or None, optional
            Training step or epoch number. Pass this to track how metrics
            change over time and plot learning curves.
        """
        pass

    @abstractmethod
    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Upload a local file to the backend's artifact store.

        Parameters
        ----------
        filename : str
            Local path to the file to upload (e.g., a saved model or config).
        artifact_id : str
            Identifier for the artifact on the backend.
        """
        pass

    @abstractmethod
    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Pickle a Python object and upload it as an artifact.

        Parameters
        ----------
        var : object
            Any picklable object (e.g., a fitted scikit-learn pipeline).
        artifact_id : str
            Identifier for the artifact on the backend.
        """
        pass

    @abstractmethod
    def load_pickle(self, artifact_id: str) -> object:
        """
        Download and unpickle an artifact that was previously saved with
        ``log_pickle``.

        Parameters
        ----------
        artifact_id : str
            Identifier of the artifact to load.

        Returns
        -------
        object
            The unpickled Python object.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """Open the run. Called automatically by the ``with`` statement."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the run. Called automatically when exiting the ``with`` block."""
        pass


class VertexAITracker(ExperimentTracker):
    """
    Experiment tracker backed by Google Cloud Vertex AI.

    Authenticates via Application Default Credentials (ADC). Either run
    ``gcloud auth application-default login``, or set
    ``GOOGLE_APPLICATION_CREDENTIALS`` to the path of a service account key
    file in your ``.env``.

    Example ``.env`` file::

        GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
        GOOGLE_CLOUD_PROJECT=my-gcp-project
        GOOGLE_CLOUD_LOCATION=us-central1
        VERTEX_STAGING_BUCKET=gs://my-bucket

    Parameters
    ----------
    experiment_name : str
        Name of the Vertex AI experiment.
    run_name : str
        Name of this run within the experiment.
    project : str or None, optional
        GCP project ID. Falls back to ``GOOGLE_CLOUD_PROJECT`` env var.
    location : str or None, optional
        GCP region (e.g. ``"us-central1"``). Falls back to
        ``GOOGLE_CLOUD_LOCATION`` env var.
    bucket_name : str or None, optional
        GCS bucket URI for artifact storage (e.g. ``"gs://my-bucket"``).
        Falls back to ``VERTEX_STAGING_BUCKET`` env var.
    mode : {"develop", "deploy"}, optional
        ``"develop"`` enables logging (default). ``"deploy"`` disables it.
    experiment_description : str or None, optional
        Short description of what this experiment is testing.
    experiment_tensorboard : bool, optional
        Whether to enable TensorBoard integration (default: ``False``).
    dotenv_path : str or None, optional
        Path to a ``.env`` file. Cannot be combined with ``project``,
        ``location``, or ``bucket_name`` if their corresponding env vars are
        also present in the file (raises ``ValueError``).
    """

    @require_dependencies("google.cloud.aiplatform", "google.cloud.storage")
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        project: str | None = None,
        location: str | None = None,
        bucket_name: str | None = None,
        mode: Literal["develop", "deploy"] = "develop",
        experiment_description: str | None = None,
        experiment_tensorboard: bool = False,
        dotenv_path: str | None = None,
    ) -> None:
        import os

        from google.cloud import aiplatform, storage

        _env_map = {
            "project": "GOOGLE_CLOUD_PROJECT",
            "location": "GOOGLE_CLOUD_LOCATION",
            "bucket_name": "VERTEX_STAGING_BUCKET",
        }
        if dotenv_path is not None:
            from dotenv import dotenv_values

            env_vals = dotenv_values(dotenv_path)
            conflicts = [p for p, e in _env_map.items() if locals()[p] is not None and e in env_vals]
            if conflicts:
                raise ValueError(f"Cannot pass both dotenv_path and direct params: {conflicts}")

        super().__init__(experiment_name, run_name, mode, dotenv_path=dotenv_path)

        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        self.bucket_name = bucket_name or os.environ.get("VERTEX_STAGING_BUCKET")
        if not self.project:
            raise ValueError("project must be provided or set as GOOGLE_CLOUD_PROJECT in .env")
        if not self.location:
            raise ValueError("location must be provided or set as GOOGLE_CLOUD_LOCATION in .env")
        if not self.bucket_name:
            raise ValueError("bucket_name must be provided or set as VERTEX_STAGING_BUCKET in .env")

        self.experiment_description = experiment_description
        self.experiment_tensorboard = experiment_tensorboard
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
        Log hyperparameters to the Vertex AI experiment run.

        Parameters
        ----------
        params : dict[str, float | int | str]
            Hyperparameter names and their values.
        """
        if self.mode == "develop":
            self.experiment_run.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the Vertex AI experiment run.

        When ``step`` is provided, uses ``log_time_series_metrics`` to record
        metrics at a specific training step (e.g., for plotting learning curves).

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            Metric names and their values.
        step : int or None, optional
            Training step or epoch number.
        """
        if self.mode == "develop":
            if step is not None:
                self.experiment_run.log_time_series_metrics(metrics, step=step)
            else:
                self.experiment_run.log_metrics(metrics)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Upload a file to GCS and register it as a Vertex AI artifact.

        Parameters
        ----------
        filename : str
            Local path to the file to upload.
        artifact_id : str
            Unique identifier for this artifact within the run.
        """
        if self.mode == "develop":
            blob = self._add_artifact(artifact_id)
            blob.upload_from_filename(filename)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Pickle an object and upload it directly to GCS (no temp file needed).

        Parameters
        ----------
        var : object
            Any picklable Python object.
        artifact_id : str
            Unique identifier for this artifact within the run.
        """
        if self.mode == "develop":
            pickle_out = pickle.dumps(var)
            blob = self._add_artifact(artifact_id)
            blob.upload_from_string(pickle_out)

    def load_pickle(self, artifact_id: str) -> object:
        """
        Download an artifact from GCS and unpickle it.

        Parameters
        ----------
        artifact_id : str
            Identifier of the artifact to load.

        Returns
        -------
        object
            The unpickled Python object.
        """
        blob = self._get_blob(artifact_id)
        pickle_in = blob.download_as_string()
        return pickle.loads(pickle_in)

    def _get_blob(self, artifact_id: str):
        """Return the GCS blob for the given artifact (does not check existence)."""
        blob_name = f"{self.experiment_name}-{self.run_name}-{artifact_id}"
        blob = self.gc_bucket.blob(blob_name)
        return blob

    def _add_artifact(self, artifact_id: str):
        """
        Register a new artifact in Vertex AI and return its GCS blob.

        Raises ``ValueError`` if the artifact already exists to prevent
        accidental overwrites.
        """
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
        """Open the Vertex AI run and execution context."""
        if self.mode == "develop":
            self.execution.__enter__()
            self.experiment_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the Vertex AI run and execution context."""
        if self.mode == "develop":
            self.execution.__exit__(exc_type, exc_value, exc_traceback)
            self.experiment_run.__exit__(exc_type, exc_value, exc_traceback)


class MLflowTracker(ExperimentTracker):
    """
    Experiment tracker backed by MLflow.

    Connects to an MLflow tracking server. For authenticated servers, embed
    credentials in the URI (e.g. ``http://user:password@host:5000``) or set
    ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` as env vars.

    The experiment is created automatically if it does not already exist.

    Example ``.env`` file::

        MLFLOW_TRACKING_URI=http://localhost:5000
        MLFLOW_TRACKING_USERNAME=my-user
        MLFLOW_TRACKING_PASSWORD=my-password
        MLFLOW_ARTIFACT_LOCATION=s3://my-bucket/artifacts

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment. Created if it does not exist.
    run_name : str
        Name of this run within the experiment.
    tracking_uri : str or None, optional
        URI of the MLflow tracking server. Falls back to
        ``MLFLOW_TRACKING_URI`` env var. Uses the local filesystem if unset.
    artifact_location : str or None, optional
        Storage location for artifacts (only applied when creating a new
        experiment). Falls back to ``MLFLOW_ARTIFACT_LOCATION`` env var.
    mode : {"develop", "deploy"}, optional
        ``"develop"`` enables logging (default). ``"deploy"`` disables it.
    tags : dict[str, str] or None, optional
        Key-value tags to attach to the run (e.g. ``{"team": "ml"}``).
    dotenv_path : str or None, optional
        Path to a ``.env`` file. Cannot be combined with ``tracking_uri`` or
        ``artifact_location`` if their env vars are also in the file.
    """

    @require_dependencies("mlflow")
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        mode: Literal["develop", "deploy"] = "develop",
        tags: dict[str, str] = None,
        dotenv_path: str | None = None,
    ) -> None:
        import os

        import mlflow

        _env_map = {"tracking_uri": "MLFLOW_TRACKING_URI", "artifact_location": "MLFLOW_ARTIFACT_LOCATION"}
        if dotenv_path is not None:
            from dotenv import dotenv_values

            env_vals = dotenv_values(dotenv_path)
            conflicts = [p for p, e in _env_map.items() if locals()[p] is not None and e in env_vals]
            if conflicts:
                raise ValueError(f"Cannot pass both dotenv_path and direct params: {conflicts}")

        super().__init__(experiment_name, run_name, mode, dotenv_path=dotenv_path)

        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.artifact_location = artifact_location or os.environ.get("MLFLOW_ARTIFACT_LOCATION")
        self.tags = tags or {}
        self._mlflow = mlflow
        self._run = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name, artifact_location=self.artifact_location)
        else:
            self.experiment_id = experiment.experiment_id

    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Log hyperparameters to the active MLflow run.

        Parameters
        ----------
        params : dict[str, float | int | str]
            Hyperparameter names and their values.
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the active MLflow run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            Metric names and their values.
        step : int or None, optional
            Training step or epoch number. Used for plotting metrics over time.
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_metrics(metrics, step=step)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Upload a file to the MLflow artifact store.

        Parameters
        ----------
        filename : str
            Local path to the file to upload.
        artifact_id : str
            Subdirectory path within the run's artifact store.
        """
        if self.mode == "develop" and self._run:
            self._mlflow.log_artifact(filename, artifact_path=artifact_id)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Pickle an object to a temp file and upload it to the MLflow artifact store.

        Parameters
        ----------
        var : object
            Any picklable Python object.
        artifact_id : str
            Name used for the ``.pkl`` file in the artifact store.
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
        Download and unpickle an artifact from the MLflow artifact store.

        Parameters
        ----------
        artifact_id : str
            Name of the artifact to load (without ``.pkl`` extension).

        Returns
        -------
        object
            The unpickled Python object.

        Raises
        ------
        RuntimeError
            If called outside a ``with`` block (no active run).
        """
        if self._run is None:
            raise RuntimeError("No active run. Use within context manager or start a run first.")

        artifact_uri = f"{self._run.info.artifact_uri}/{artifact_id}.pkl"
        local_path = self._mlflow.artifacts.download_artifacts(artifact_uri)
        with open(local_path, "rb") as f:
            return pickle.load(f)

    def __enter__(self):
        """Start a new MLflow run."""
        if self.mode == "develop":
            self._run = self._mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                tags=self.tags,
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """End the active MLflow run."""
        if self.mode == "develop" and self._run:
            self._mlflow.end_run()
            self._run = None


class WandbTracker(ExperimentTracker):
    """
    Experiment tracker backed by Weights & Biases (wandb).

    Get your API key at https://wandb.ai/authorize and set it in a ``.env``
    file before initializing this tracker.

    Example ``.env`` file::

        WANDB_API_KEY=your_api_key_here
        WANDB_ENTITY=my-team

    Parameters
    ----------
    experiment_name : str
        Name of the wandb project. Runs are grouped under this project.
    run_name : str
        Display name for this run in the wandb UI.
    mode : {"develop", "deploy"}, optional
        ``"develop"`` enables logging (default). ``"deploy"`` disables it.
    entity : str or None, optional
        Your wandb username or team name. Falls back to ``WANDB_ENTITY`` env var.
    tags : list[str] or None, optional
        Tags for filtering and searching runs (e.g. ``["baseline", "v2"]``).
    config : dict or None, optional
        Initial hyperparameters to record in the run config. Can also be
        updated later with ``log_params``.
    dotenv_path : str or None, optional
        Path to a ``.env`` file. Cannot be combined with ``entity`` if
        ``WANDB_ENTITY`` is also present in the file.
    """

    @require_dependencies("wandb")
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        mode: Literal["develop", "deploy"] = "develop",
        entity: str | None = None,
        tags: list[str] = None,
        config: dict = None,
        dotenv_path: str | None = None,
    ) -> None:
        import os

        import wandb

        if dotenv_path is not None and entity is not None:
            from dotenv import dotenv_values

            if "WANDB_ENTITY" in dotenv_values(dotenv_path):
                raise ValueError("Cannot pass both dotenv_path and entity: use one or the other")

        super().__init__(experiment_name, run_name, mode, dotenv_path=dotenv_path)
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        self.tags = tags
        self.config = config or {}
        self._wandb = wandb
        self._run = None

    def log_params(self, params: dict[str, float | int | str]) -> None:
        """
        Update the wandb run config with hyperparameters.

        Parameters
        ----------
        params : dict[str, float | int | str]
            Hyperparameter names and their values.
        """
        if self.mode == "develop" and self._run:
            self._run.config.update(params)

    def log_metrics(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        """
        Log metrics to the active wandb run.

        Parameters
        ----------
        metrics : dict[str, float | int | str]
            Metric names and their values.
        step : int or None, optional
            Training step number. Used for the x-axis in wandb charts.
        """
        if self.mode == "develop" and self._run:
            self._run.log(metrics, step=step)

    def log_file(self, filename: str, artifact_id: str) -> None:
        """
        Upload a file to wandb as an artifact.

        Parameters
        ----------
        filename : str
            Local path to the file to upload.
        artifact_id : str
            Name for the artifact in wandb.
        """
        if self.mode == "develop" and self._run:
            artifact = self._wandb.Artifact(artifact_id, type="file")
            artifact.add_file(filename)
            self._run.log_artifact(artifact)

    def log_pickle(self, var: object, artifact_id: str) -> None:
        """
        Pickle an object to a temp file and upload it to wandb as an artifact.

        Parameters
        ----------
        var : object
            Any picklable Python object.
        artifact_id : str
            Name for the artifact in wandb.
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
        Download the latest version of an artifact from wandb and unpickle it.

        Parameters
        ----------
        artifact_id : str
            Name of the artifact to load.

        Returns
        -------
        object
            The unpickled Python object.

        Raises
        ------
        RuntimeError
            If called outside a ``with`` block (no active run).
        """
        if self._run is None:
            raise RuntimeError("No active run. Use within context manager or start a run first.")

        artifact = self._run.use_artifact(f"{artifact_id}:latest")
        artifact_dir = artifact.download()
        filepath = f"{artifact_dir}/{artifact_id}.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __enter__(self):
        """Initialize a new wandb run."""
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
        """Finish the wandb run and sync all data."""
        if self.mode == "develop" and self._run:
            self._run.finish()
            self._run = None


def create_tracker(
    backend: Literal["vertex_ai", "mlflow", "wandb"],
    experiment_name: str,
    run_name: str,
    mode: Literal["develop", "deploy"] = "develop",
    dotenv_path: str | None = None,
    **kwargs,
) -> ExperimentTracker:
    """
    Factory function to create an experiment tracker for the given backend.

    Prefer this function over instantiating tracker classes directly, as it
    keeps backend-specific imports isolated.

    Parameters
    ----------
    backend : {"vertex_ai", "mlflow", "wandb"}
        The experiment tracking backend to use.
    experiment_name : str
        Name of the experiment.
    run_name : str
        Name of this run within the experiment.
    mode : {"develop", "deploy"}, optional
        ``"develop"`` enables logging (default). ``"deploy"`` disables it.
    dotenv_path : str or None, optional
        Path to a ``.env`` file containing credentials and config. Cannot be
        combined with a kwarg whose corresponding env var is also in the file.
    **kwargs
        Backend-specific arguments forwarded to the tracker constructor.

        For ``"vertex_ai"``:
            - ``project`` (str) — GCP project ID (env: ``GOOGLE_CLOUD_PROJECT``)
            - ``location`` (str) — GCP region, e.g. ``"us-central1"``
              (env: ``GOOGLE_CLOUD_LOCATION``)
            - ``bucket_name`` (str) — GCS bucket URI
              (env: ``VERTEX_STAGING_BUCKET``)
            - ``experiment_description`` (str) — Short experiment description
            - ``experiment_tensorboard`` (bool) — Enable TensorBoard
            - Credentials: ``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``

        For ``"mlflow"``:
            - ``tracking_uri`` (str) — Tracking server URI
              (env: ``MLFLOW_TRACKING_URI``)
            - ``artifact_location`` (str) — Artifact storage path
              (env: ``MLFLOW_ARTIFACT_LOCATION``)
            - ``tags`` (dict[str, str]) — Run tags
            - Credentials: ``MLFLOW_TRACKING_USERNAME`` /
              ``MLFLOW_TRACKING_PASSWORD`` in ``.env``

        For ``"wandb"``:
            - ``entity`` (str) — wandb username or team
              (env: ``WANDB_ENTITY``)
            - ``tags`` (list[str]) — Run tags
            - ``config`` (dict) — Initial hyperparameters
            - Credentials: ``WANDB_API_KEY`` in ``.env``

    Returns
    -------
    ExperimentTracker
        A ready-to-use tracker instance. Use it as a context manager.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported values.

    Examples
    --------
    >>> tracker = create_tracker(
    ...     backend="wandb",
    ...     experiment_name="credit-model",
    ...     run_name="xgb-v1",
    ...     dotenv_path=".env",
    ... )
    >>> with tracker:
    ...     tracker.log_params({"learning_rate": 0.01, "max_depth": 6})
    ...     tracker.log_metrics({"auc": 0.87}, step=1)

    >>> tracker = create_tracker(
    ...     backend="mlflow",
    ...     experiment_name="credit-model",
    ...     run_name="xgb-v1",
    ...     tracking_uri="http://localhost:5000",
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
            dotenv_path=dotenv_path,
        )
    elif backend == "mlflow":
        return MLflowTracker(
            experiment_name=experiment_name,
            run_name=run_name,
            mode=mode,
            tracking_uri=kwargs.get("tracking_uri"),
            artifact_location=kwargs.get("artifact_location"),
            tags=kwargs.get("tags"),
            dotenv_path=dotenv_path,
        )
    elif backend == "wandb":
        return WandbTracker(
            experiment_name=experiment_name,
            run_name=run_name,
            mode=mode,
            entity=kwargs.get("entity"),
            tags=kwargs.get("tags"),
            config=kwargs.get("config"),
            dotenv_path=dotenv_path,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'vertex_ai', 'mlflow', or 'wandb'.")
