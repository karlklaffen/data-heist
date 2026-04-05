import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RegressionModel:
    def __init__(self, model_type, name='Model', use_scaler=False):
        """
        Initialize a regression model.
        Args:
            model_type: 'linear', 'ridge', or 'random_forest'
            name: Name identifier for the model
            use_scaler: Whether to apply StandardScaler to features (default: False)
        """
        self.name = name
        self.model_type = model_type
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(self, X_train, y_train):
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        if self.scaler:
            X_scaled = self.scaler.transform(X_test)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X_test)

