from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class MulCmdModels:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.poly_features = None
        self.X_poly_train = None
        self.X_poly_test = None

    def set_model(self, model_name, X_train=None, y_train=None, X_test=None, y_test=None, **kwargs):
        models = {
            'linear_regression': (LinearRegression, 'Linear Regression'),
            'ridge_regression': (Ridge, 'Ridge Regression'),
            'lasso_regression': (Lasso, 'Lasso Regression'),
            'svr': (SVR, 'Support Vector Regression'),
            'svc': (SVC, 'Support Vector Classification'),
            'polynomial_regression': (self.create_polynomial_regression(kwargs.get('degree', 2)), 'Polynomial Regression'),
            'knn': (KNeighborsClassifier, 'K-Nearest Neighbors'),
            'naive_bayes': (GaussianNB, 'Naive Bayes (Gaussian)')
        }

        model_info = models.get(model_name.lower())
        if model_info is None:
            return None, f"Error: Model '{model_name}' is not recognized or not available in MulCmdModels", None, None, None, None

        model_class, model_full_name = model_info

        # For polynomial regression, the model is created as a pipeline and can't accept kwargs directly
        if model_name.lower() == 'polynomial_regression':
            model = model_class
            self.poly_features = model.named_steps['polynomialfeatures']
            self.X_poly_train = self.poly_features.fit_transform(X_train)
            self.X_poly_test = self.poly_features.transform(X_test)
            message = f"Model '{model_full_name}' selected \nPolynomial features degree: {self.poly_features.degree}"
        else:
            # Create the model with kwargs
            model_params = {k: v for k, v in kwargs.items() if k in model_class().get_params()}
            model = model_class(**model_params)
            message = f"Model '{model_full_name}' selected \nHyper parameters: {model}"

        self.model_name = model_name.lower()
        self.model = model

        # Return model, message, and data for training/testing
        return self.model, message, X_train, y_train, X_test, y_test

    def create_polynomial_regression(self, degree):
        return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
