"""Tests for experiment tracker module."""

import sys
from unittest.mock import MagicMock

import pytest


class TestExperimentTrackerBase:
    """Tests for ExperimentTracker abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that ExperimentTracker cannot be instantiated directly."""
        from src.caketool.experiment import ExperimentTracker

        with pytest.raises(TypeError):
            ExperimentTracker("exp", "run")

    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        from src.caketool.experiment import ExperimentTracker

        abstract_methods = {
            "log_params",
            "log_metrics",
            "log_file",
            "log_pickle",
            "load_pickle",
            "__enter__",
            "__exit__",
        }
        assert abstract_methods.issubset(set(ExperimentTracker.__abstractmethods__))


class TestCreateTrackerFactory:
    """Tests for create_tracker factory function."""

    def test_create_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError."""
        from src.caketool.experiment import create_tracker

        with pytest.raises(ValueError, match="Unknown backend"):
            create_tracker(
                backend="unknown",
                experiment_name="test-exp",
                run_name="run-001",
            )


@pytest.fixture
def mock_mlflow_module():
    """Set up mock mlflow module in sys.modules."""
    mock_mlflow = MagicMock()
    mock_mlflow.get_experiment_by_name.return_value = None
    mock_mlflow.create_experiment.return_value = "exp-123"
    mock_run = MagicMock()
    mock_run.info.artifact_uri = "file:///tmp/artifacts"
    mock_mlflow.start_run.return_value = mock_run

    original_modules = sys.modules.copy()
    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = MagicMock()

    yield mock_mlflow

    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


@pytest.fixture
def mock_gcp_modules():
    """Set up mock GCP modules in sys.modules."""
    mock_aiplatform = MagicMock()
    mock_run = MagicMock()
    mock_aiplatform.start_run.return_value = mock_run
    mock_aiplatform.start_execution.return_value = MagicMock()

    mock_storage = MagicMock()
    mock_bucket = MagicMock()
    mock_storage.Client.return_value = MagicMock()
    mock_storage.Bucket.return_value = mock_bucket

    original_modules = sys.modules.copy()

    # Set up google cloud modules
    mock_google = MagicMock()
    mock_google_cloud = MagicMock()
    sys.modules["google"] = mock_google
    sys.modules["google.cloud"] = mock_google_cloud
    sys.modules["google.cloud.aiplatform"] = mock_aiplatform
    sys.modules["google.cloud.storage"] = mock_storage

    yield {"aiplatform": mock_aiplatform, "storage": mock_storage, "run": mock_run}

    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


class TestMLflowTrackerUnit:
    """Unit tests for MLflowTracker class."""

    def test_init_creates_new_experiment(self, mock_mlflow_module):
        """Test MLflowTracker initialization with new experiment."""
        # Need to reload module to pick up mocked mlflow
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_mlflow_module.get_experiment_by_name.return_value = None
        mock_mlflow_module.create_experiment.return_value = "new-exp-123"

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            tracking_uri="http://localhost:5000",
        )

        assert tracker.experiment_name == "test-exp"
        assert tracker.run_name == "run-001"
        assert tracker.experiment_id == "new-exp-123"
        mock_mlflow_module.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_init_uses_existing_experiment(self, mock_mlflow_module):
        """Test MLflowTracker initialization with existing experiment."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing-exp-456"
        mock_mlflow_module.get_experiment_by_name.return_value = mock_experiment

        tracker = tracker_module.MLflowTracker(
            experiment_name="existing-exp",
            run_name="run-001",
        )

        assert tracker.experiment_id == "existing-exp-456"
        mock_mlflow_module.create_experiment.assert_not_called()

    def test_context_manager_develop_mode(self, mock_mlflow_module):
        """Test MLflowTracker context manager in develop mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        with tracker:
            mock_mlflow_module.start_run.assert_called_once()
            assert tracker._run == mock_run

        mock_mlflow_module.end_run.assert_called_once()
        assert tracker._run is None

    def test_context_manager_deploy_mode_no_run(self, mock_mlflow_module):
        """Test MLflowTracker context manager in deploy mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="deploy",
        )

        with tracker:
            mock_mlflow_module.start_run.assert_not_called()
            assert tracker._run is None

    def test_log_params_in_develop_mode(self, mock_mlflow_module):
        """Test logging parameters within context manager."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        params = {"learning_rate": 0.01, "batch_size": 32}
        with tracker:
            tracker.log_params(params)
            mock_mlflow_module.log_params.assert_called_once_with(params)

    def test_log_params_in_deploy_mode_no_op(self, mock_mlflow_module):
        """Test logging parameters in deploy mode is a no-op."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="deploy",
        )

        with tracker:
            tracker.log_params({"param": 1})
            mock_mlflow_module.log_params.assert_not_called()

    def test_log_metrics_without_step(self, mock_mlflow_module):
        """Test logging metrics without step."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        metrics = {"accuracy": 0.95, "f1": 0.92}
        with tracker:
            tracker.log_metrics(metrics)
            mock_mlflow_module.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_metrics_with_step(self, mock_mlflow_module):
        """Test logging metrics with step."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        metrics = {"loss": 0.1}
        with tracker:
            tracker.log_metrics(metrics, step=5)
            mock_mlflow_module.log_metrics.assert_called_once_with(metrics, step=5)

    def test_log_file(self, mock_mlflow_module):
        """Test logging a file as artifact."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        with tracker:
            tracker.log_file("/path/to/file.csv", "data")
            mock_mlflow_module.log_artifact.assert_called_once_with("/path/to/file.csv", artifact_path="data")

    def test_load_pickle_without_run_raises(self, mock_mlflow_module):
        """Test that load_pickle raises error without active run."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            mode="develop",
        )

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.load_pickle("model")

    def test_tags_passed_to_run(self, mock_mlflow_module):
        """Test that tags are passed to the run."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tags = {"version": "1.0", "author": "test"}
        tracker = tracker_module.MLflowTracker(
            experiment_name="test-exp",
            run_name="run-001",
            tags=tags,
        )

        with tracker:
            mock_mlflow_module.start_run.assert_called_once_with(
                experiment_id="exp-123",
                run_name="run-001",
                tags=tags,
            )


class TestVertexAITrackerUnit:
    """Unit tests for VertexAITracker class."""

    def test_init_develop_mode(self, mock_gcp_modules):
        """Test VertexAITracker initialization in develop mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="develop",
        )

        assert tracker.experiment_name == "test-exp"
        assert tracker.run_name == "run-001"
        assert tracker.mode == "develop"
        assert tracker.project == "test-project"
        assert tracker.bucket_name == "test-bucket"

    def test_init_deploy_mode(self, mock_gcp_modules):
        """Test VertexAITracker initialization in deploy mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="deploy",
        )

        assert tracker.mode == "deploy"

    def test_log_params_develop_mode(self, mock_gcp_modules):
        """Test logging parameters in develop mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="develop",
        )

        params = {"learning_rate": 0.01, "n_estimators": 100}
        tracker.log_params(params)

        # Assert on the tracker's own experiment_run mock
        tracker.experiment_run.log_params.assert_called_once_with(params)

    def test_log_params_deploy_mode_no_op(self, mock_gcp_modules):
        """Test that logging parameters in deploy mode is a no-op."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="deploy",
        )

        # Should not raise any error
        tracker.log_params({"param": 1})

    def test_log_metrics_without_step(self, mock_gcp_modules):
        """Test logging metrics without step."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="develop",
        )

        metrics = {"accuracy": 0.95, "loss": 0.05}
        tracker.log_metrics(metrics)

        # Assert on the tracker's own experiment_run mock
        tracker.experiment_run.log_metrics.assert_called_once_with(metrics)

    def test_log_metrics_with_step(self, mock_gcp_modules):
        """Test logging metrics with step for time-series tracking."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.VertexAITracker(
            project="test-project",
            location="us-central1",
            experiment_name="test-exp",
            run_name="run-001",
            bucket_name="test-bucket",
            mode="develop",
        )

        metrics = {"accuracy": 0.95}
        tracker.log_metrics(metrics, step=10)

        # Assert on the tracker's own experiment_run mock
        tracker.experiment_run.log_time_series_metrics.assert_called_once_with(metrics, step=10)


@pytest.fixture
def mock_wandb_module():
    """Set up mock wandb module in sys.modules."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    original_modules = sys.modules.copy()
    sys.modules["wandb"] = mock_wandb

    yield mock_wandb, mock_run

    sys.modules.clear()
    sys.modules.update(original_modules)


class TestWandbTrackerUnit:
    """Unit tests for WandbTracker class."""

    def test_init(self, mock_wandb_module):
        """Test WandbTracker initialization."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, _ = mock_wandb_module
        tracker = tracker_module.WandbTracker(
            experiment_name="my-project",
            run_name="run-001",
            entity="my-team",
            tags=["v1"],
        )

        assert tracker.experiment_name == "my-project"
        assert tracker.run_name == "run-001"
        assert tracker.entity == "my-team"
        assert tracker.tags == ["v1"]

    def test_context_manager_develop_mode(self, mock_wandb_module):
        """Test WandbTracker context manager calls wandb.init and finish."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, mock_run = mock_wandb_module
        tracker = tracker_module.WandbTracker(
            experiment_name="my-project",
            run_name="run-001",
            mode="develop",
        )

        with tracker:
            mock_wandb.init.assert_called_once_with(
                project="my-project",
                name="run-001",
                entity=None,
                tags=None,
                config={},
            )
            assert tracker._run == mock_run

        mock_run.finish.assert_called_once()
        assert tracker._run is None

    def test_context_manager_deploy_mode_no_run(self, mock_wandb_module):
        """Test WandbTracker context manager in deploy mode does not init."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, _ = mock_wandb_module
        tracker = tracker_module.WandbTracker(
            experiment_name="my-project",
            run_name="run-001",
            mode="deploy",
        )

        with tracker:
            mock_wandb.init.assert_not_called()
            assert tracker._run is None

    def test_log_params(self, mock_wandb_module):
        """Test logging params updates run config."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, mock_run = mock_wandb_module
        tracker = tracker_module.WandbTracker(experiment_name="proj", run_name="run-001")

        params = {"lr": 0.01, "depth": 6}
        with tracker:
            tracker.log_params(params)
            mock_run.config.update.assert_called_once_with(params)

    def test_log_metrics(self, mock_wandb_module):
        """Test logging metrics calls run.log."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, mock_run = mock_wandb_module
        tracker = tracker_module.WandbTracker(experiment_name="proj", run_name="run-001")

        metrics = {"accuracy": 0.95}
        with tracker:
            tracker.log_metrics(metrics, step=3)
            mock_run.log.assert_called_once_with(metrics, step=3)

    def test_log_params_deploy_mode_no_op(self, mock_wandb_module):
        """Test that logging in deploy mode is a no-op."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, mock_run = mock_wandb_module
        tracker = tracker_module.WandbTracker(experiment_name="proj", run_name="run-001", mode="deploy")

        with tracker:
            tracker.log_params({"param": 1})
            tracker.log_metrics({"metric": 0.5})
            mock_run.config.update.assert_not_called()
            mock_run.log.assert_not_called()

    def test_load_pickle_without_run_raises(self, mock_wandb_module):
        """Test that load_pickle raises error without active run."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.WandbTracker(experiment_name="proj", run_name="run-001")

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.load_pickle("model")


class TestCreateTrackerIntegration:
    """Integration tests for create_tracker factory function."""

    def test_create_mlflow_tracker(self, mock_mlflow_module):
        """Test creating MLflow tracker via factory."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.create_tracker(
            backend="mlflow",
            experiment_name="test-exp",
            run_name="run-001",
            tracking_uri="http://localhost:5000",
        )

        assert isinstance(tracker, tracker_module.MLflowTracker)
        assert tracker.experiment_name == "test-exp"
        assert tracker.tracking_uri == "http://localhost:5000"

    def test_create_vertex_ai_tracker(self, mock_gcp_modules):
        """Test creating VertexAI tracker via factory."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.create_tracker(
            backend="vertex_ai",
            experiment_name="test-exp",
            run_name="run-001",
            project="test-project",
            location="us-central1",
            bucket_name="test-bucket",
        )

        assert isinstance(tracker, tracker_module.VertexAITracker)
        assert tracker.experiment_name == "test-exp"
        assert tracker.project == "test-project"

    def test_create_wandb_tracker(self, mock_wandb_module):
        """Test creating wandb tracker via factory."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_wandb, _ = mock_wandb_module
        tracker = tracker_module.create_tracker(
            backend="wandb",
            experiment_name="my-project",
            run_name="run-001",
            entity="my-team",
        )

        assert isinstance(tracker, tracker_module.WandbTracker)
        assert tracker.experiment_name == "my-project"
        assert tracker.entity == "my-team"

    def test_create_tracker_with_deploy_mode(self, mock_mlflow_module):
        """Test creating tracker in deploy mode."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.create_tracker(
            backend="mlflow",
            experiment_name="test-exp",
            run_name="run-001",
            mode="deploy",
        )

        assert tracker.mode == "deploy"


class TestTrackerWorkflow:
    """End-to-end workflow tests for trackers."""

    def test_full_mlflow_workflow(self, mock_mlflow_module):
        """Test complete MLflow tracking workflow."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        mock_run = MagicMock()
        mock_mlflow_module.start_run.return_value = mock_run

        tracker = tracker_module.create_tracker(
            backend="mlflow",
            experiment_name="xgboost-experiment",
            run_name="tuning-run-001",
            tags={"model": "xgboost"},
        )

        with tracker:
            # Log hyperparameters
            tracker.log_params(
                {
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "n_estimators": 100,
                }
            )

            # Log metrics at different steps
            for epoch in range(3):
                tracker.log_metrics({"train_loss": 0.5 - epoch * 0.1}, step=epoch)

        # Verify calls
        assert mock_mlflow_module.log_params.call_count == 1
        assert mock_mlflow_module.log_metrics.call_count == 3

    def test_deploy_mode_no_logging(self, mock_mlflow_module):
        """Test that deploy mode doesn't log anything."""
        import importlib

        import src.caketool.experiment.experiment_tracker as tracker_module

        importlib.reload(tracker_module)

        tracker = tracker_module.create_tracker(
            backend="mlflow",
            experiment_name="prod-model",
            run_name="inference-001",
            mode="deploy",
        )

        with tracker:
            tracker.log_params({"param": 1})
            tracker.log_metrics({"metric": 0.5})
            tracker.log_file("/path/file.txt", "artifacts")

        # No logging calls in deploy mode
        mock_mlflow_module.start_run.assert_not_called()
        mock_mlflow_module.log_params.assert_not_called()
        mock_mlflow_module.log_metrics.assert_not_called()
        mock_mlflow_module.log_artifact.assert_not_called()
