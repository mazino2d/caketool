"""Model building utilities for credit risk and machine learning tasks.

Provides XGBoost-based models with integrated preprocessing pipelines for
binary classification, multiclass classification, ranking, and regression.

Task-specific models
--------------------
- ``BinaryBoostTree`` – binary classification (``binary:logistic``)
- ``MulticlassBoostTree`` – multiclass classification (``multi:softprob``)
- ``BoostRanker`` – learning-to-rank (``rank:ndcg``)
- ``BoostRegressor`` – regression (``reg:squarederror``)

Ensemble utilities
------------------
- ``BaseEnsemble`` – averages predictions from a list of ``BaseBoostTree`` models
- ``VotingModel`` – averages predictions from any sklearn-compatible estimators

Configuration
-------------
- ``ModelConfig`` – typed dataclass for all model and preprocessing parameters
"""

from .base.boost_tree import BaseBoostTree as BaseBoostTree
from .base.ensemble import BaseEnsemble as BaseEnsemble
from .base.voting_model import VotingModel as VotingModel
from .classification.binary import BinaryBoostTree as BinaryBoostTree
from .classification.multiclass import MulticlassBoostTree as MulticlassBoostTree
from .config import ModelConfig as ModelConfig
from .ranking.ranker import BoostRanker as BoostRanker
from .regression.regressor import BoostRegressor as BoostRegressor
