import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from Logger import CustomLogger

logs = CustomLogger()


class clean:
    """
    This class object handles regression problems
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.max_scaler = MinMaxScaler()
        self.le_encoder = LabelEncoder()
        self.numerical_cols = None
        self.categorical_cols = None
        self.column_transformer = None
        self.fit_status = False
        self.preprocessing_pipeline = None
        self.fit_status = False
        self.feature_names_out = None  # Store feature names after transformation

    def correlation_multicollinearity(self, X, threshold=0.9):
        """
        Checks for multi-collinearity between features using pearson correlation
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features = X.select_dtypes(include=[float, int]).columns.tolist()
            x_num = X[numeric_features].dropna()
            correlation_matrix = x_num.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_correlation_pairs = [(column, row) for row in upper_triangle.index for column in upper_triangle.columns
                                      if upper_triangle.loc[row, column] > threshold]
            columns_to_drop = {column for column, row in high_correlation_pairs}
            df_reduced = X.drop(columns=columns_to_drop)
            logs.log("Successfully dropped mult-collinear columns!")
            return df_reduced.columns, columns_to_drop
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logs.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    # def preprocessor_fit(self, X, one_hot_encode_cols=None, label_encode_cols=None):
    #     """
    #     Fit the preprocessor on the data.
    #
    #     Args:
    #         X (pd.DataFrame): Input data containing both numerical and categorical columns.
    #         one_hot_encode_cols (list): List of categorical columns to one-hot encode.
    #         label_encode_cols (list): List of categorical columns to label encode.
    #     """
    #     try:
    #         self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    #         self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    #
    #         transformers = []
    #
    #         if self.numerical_cols:
    #             num_pipeline = Pipeline([
    #                 ('num_imputer', SimpleImputer(strategy='mean')),
    #                 ('scaler', StandardScaler())
    #             ])
    #             transformers.append(('num', num_pipeline, self.numerical_cols))
    #
    #         if self.categorical_cols:
    #             if one_hot_encode_cols:
    #                 cat_pipeline = Pipeline([
    #                     ('cat_imputer', SimpleImputer(strategy='most_frequent')),
    #                     ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #                 ])
    #                 transformers.append(('cat_onehot', cat_pipeline, one_hot_encode_cols))
    #
    #             if label_encode_cols:
    #                 for col in label_encode_cols:
    #                     transformers.append((f'{col}_label', FunctionTransformer(self.label_encode), [col]))
    #
    #         self.preprocessing_pipeline = ColumnTransformer(transformers, remainder='passthrough')
    #         self.preprocessing_pipeline.fit(X)
    #         self.fit_status = True
    #         self.feature_names_out = self.get_feature_names_out()
    #         logs.log("Successfully fit the pre-processing pipeline!")
    #     except Exception as e:
    #         raise ValueError(f"Something went wrong while running preprocessor_fit method: {e}")
    #         logs.log(f"Error during fit: {str(e)}")

    # def preprocessor_transform(self, X):
    #     """
    #     Transform the input data using the fitted preprocessor.
    #
    #     Args:
    #         X (pd.DataFrame): Input data to transform.
    #
    #     Returns:
    #         pd.DataFrame: Transformed data with original column names.
    #     """
    #     try:
    #         if not self.fit_status:
    #             raise ValueError("Preprocessor must be fit on data before transforming.")
    #         transformed_data = self.preprocessing_pipeline.transform(X)
    #         transformed_df = pd.DataFrame(transformed_data, columns=self.feature_names_out)
    #         logs.log("Successfully transformed the dataset using the pre-processing pipeline")
    #         return transformed_df
    #     except Exception as e:
    #         raise ValueError(f"Something went wrong while running preprocessor_transform method: {e}")
    #         logs.log(f"Error during transform: {str(e)}")

    def get_feature_names_out(self):
        """
        Get feature names after transformation.

        Returns:
            list: List of feature names after transformation.
        """
        try:
            if self.preprocessing_pipeline is None:
                return []

            feature_names_out = []
            for name, trans, column_names in self.preprocessing_pipeline.transformers_:
                if trans == 'drop' or trans == 'passthrough':
                    continue
                if isinstance(trans, Pipeline):
                    if name.startswith('cat_onehot') and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_out.extend(trans.named_steps['onehot'].get_feature_names_out())
                    else:
                        feature_names_out.extend(column_names)
                elif isinstance(trans, FunctionTransformer):
                    feature_names_out.extend(column_names)
                else:
                    feature_names_out.extend(column_names)
            logs.log("Successfully retrieved features name!")
            return feature_names_out

        except Exception as e:
            raise ValueError(f"Something went wrong while running get_feature_names_out method: {e}")
            logs.log(f"Error during get_feature_names_out: {str(e)}")

    def label_encode(self, X):
        """
        Apply label encoding to the input data.

        Args:
            X (pd.Series or pd.DataFrame): Input data to encode.

        Returns:
            np.ndarray: Label encoded data reshaped to 2D.
        """
        try:
            # Ensure that X is a DataFrame with a single column
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]
            elif isinstance(X, pd.Series):
                X = X
            else:
                raise ValueError("Input to label_encode must be a pandas DataFrame or Series")

            # Debugging: Print the input shape and head
            #print(f"Label encoding input shape: {X.shape}")
            #print(f"Label encoding input head:\n{X.head()}")
            le = LabelEncoder()
            encoded = le.fit_transform(X.squeeze())
            # Debugging: Print the encoded output
            #print(f"Encoded output: {encoded}")
            #le = LabelEncoder()
            logs.log("Successfully performed label encoding!")
            return encoded.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Something went wrong while performing label encoding: {e}")
            logs.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def find_numericategory_columns(self, X):
        """
        Find numerical and categorical columns from dataset
        :param X:
        :return: numeric and categorical column names
        """
        try:
            numeric_cols = X.select_dtypes(include=[float, int]).columns.tolist()
            categoric_cols = X.select_dtypes(exclude=[float, int]).columns.tolist()
            logs.log("Successfully extracted numerical and categorical columns!")
            return numeric_cols, categoric_cols
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the numeric and categorical columns: {e}")
            logs.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def plot_confusion_matrix(self, y_pred, y):
        try:

            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix')
            plt.savefig(f'confusion_matrix.png')
            logs.log("Successfully plotted the confusion matrix!")
        except Exception as e:
            raise ValueError(f"Error while plotting the confusion matrix : {e}")
            logs.log("An error was raised while plotting the confusion matrix", level='ERROR')

    def plot_roc_curve(self, fpr, tpr):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.savefig(f'roc_auc.png')
            logs.log("Successfully plotted the roc-auc curve!")
        except Exception as e:
            raise ValueError(f"Error while plotting the Receiver Operating Characteristic (ROC) Curve : {e}")
            logs.log("An error was raised while plotting the Receiver Operating Characteristic (ROC) Curve",
                       level='ERROR')

    def plot_class_distribution(self, df, target):
        """
        Plots the distribution of the target variable.

        Parameters:
        - df: DataFrame, the input dataframe containing the target variable.
        - target: str, the name of the target variable column.

        Returns:
        - None
        """
        # Plot count distribution
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.countplot(x=target, data=df)
            plt.title('Count Plot: Class Distribution of Target Variable')
            plt.xlabel('Class')
            plt.ylabel('Count')

            # Plot pie chart
            plt.subplot(1, 2, 2)
            df[target].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
            plt.title('Pie Plot : Class Distribution of Target Variable')
            plt.ylabel('')

            plt.tight_layout()
            plt.savefig(f'confusion_matrix.png')
            logs.log("Successfully rendered distribution plots of target variable")
        except Exception as e:
            raise ValueError(f"Error while rendering distribution plots of target variable : {e}")
            logs.log("An error was raised while rendering distribution plots of target variable", level='ERROR')

    def split_train_test(self, X, y, test_size=0.2):
        """
        Split dataset into tain and test split using test size of 20%
        :param X:
        :param y:
        :param test_size:
        :return:
        """
        try:
            # Ensure the data is sorted by date
            x = X.sort_index()
            # Calculate the number of test samples
            n_test = int(len(x) * test_size)
            # Split the data
            x_train = x[:-n_test]
            y_train = y[:-n_test]
            x_test = x[-n_test:]
            y_test = y[-n_test:]
            #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            logs.log("Successfully split the dataset to train-test set!")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise ValueError(f"Error in splitting dataset into train-test sets {e}")
            logs.log("Something went wrong while splitting dataset into train-test sets", level='ERROR')

    # def get_feature_names_out(self):
    #     if self.preprocessing_pipeline is not None:
    #         return self.preprocessing_pipeline.get_feature_names_out()
    #     return []

    def preprocessor_fit(self, X, one_hot_encode_cols=None, label_encode_cols=None):
        """
        Fit the preprocessor on the data.

        Args:
            X (pd.DataFrame): Input data containing both numerical and categorical columns.
            one_hot_encode_cols (list): List of categorical columns to one-hot encode.
            label_encode_cols (list): List of categorical columns to label encode.
        """
        try:
            self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            transformers = []

            if self.numerical_cols:
                num_pipeline = Pipeline([
                    ('num_imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', num_pipeline, self.numerical_cols))

            if self.categorical_cols:
                if one_hot_encode_cols:
                    cat_pipeline = Pipeline([
                        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])
                    transformers.append(('cat_onehot', cat_pipeline, one_hot_encode_cols))

                if label_encode_cols:
                    for col in label_encode_cols:
                        transformers.append((f'{col}_label', FunctionTransformer(self.label_encode), [col]))

            self.preprocessing_pipeline = ColumnTransformer(transformers, remainder='passthrough')
            self.preprocessing_pipeline.fit(X)
            self.fit_status = True
            self.feature_names_out = self.get_feature_names_out()
            logs.log("Successfully fit the pre-processing pipeline!")
        except Exception as e:
            raise ValueError(f"Something went wrong while running preprocessor_fit method: {e}")
            logs.log(f"Error during fit: {str(e)}")

    def preprocessor_transform(self, X):
        """
        Transform the input data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data with original column names.
        """
        try:
            if not self.fit_status:
                raise ValueError("Preprocessor must be fit on data before transforming.")

            # Debugging: Print the input data shape and head
            #print(f"Transforming data with shape: {X.shape}")
            #print(f"Transforming data head:\n{X.head()}")

            transformed_data = self.preprocessing_pipeline.transform(X)

            # Debugging: Print the shape of transformed data
            #print(f"Transformed data shape: {transformed_data.shape}")

            transformed_df = pd.DataFrame(transformed_data, columns=self.feature_names_out)

            # Debugging: Print the head of the transformed dataframe
            #print(f"Transformed data head:\n{transformed_df.head()}")

            logs.log("Successfully transformed the dataset using the pre-processing pipeline")
            return transformed_df
        except Exception as e:
            raise ValueError(f"Something went wrong while running preprocessor_transform method: {e}")

    def __str__(self):
        return "This is my custom feature engineering class object"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Installed successfully!")
