from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class BaseMMModel(ABC):
    """Abstract base class for Market Mix Models."""

    def __init__(self):
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        media_columns: List[str],
        control_columns: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit the Market Mix Model.

        Args:
            data: Input DataFrame
            target: Target variable column name
            media_columns: List of media spend columns
            control_columns: Optional list of control variables
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            data: Input DataFrame for predictions

        Returns:
            Numpy array of predictions
        """
        pass

    @abstractmethod
    def get_channel_contributions(self) -> Dict[str, float]:
        """
        Get contribution of each media channel.

        Returns:
            Dictionary of channel contributions
        """
        pass

    @abstractmethod
    def get_model_summary(self) -> Dict[str, Union[float, Dict]]:
        """
        Get comprehensive model summary.

        Returns:
            Dictionary of model performance metrics
        """
        pass
