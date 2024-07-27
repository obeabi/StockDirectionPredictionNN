import numpy as np

from stocks import instrument
import pandas as pd
import matplotlib.pyplot as plt
from ExploratoryAnalysis import extensive_eda
from FeatureEngineering import clean
from ClassifierNN import ANNClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from keras.models import load_model

# Object of class stocks
ticker = 'AAPL'
asset = instrument(ticker=ticker, interval='1d')
# Object of class MLClassifier

# Collect the dataset
# df = asset.download_price_volume()
df = asset.add_technical_indicators()
X = df.drop(columns=['Target'])
y = df['Target']
# Perform EDA using extensive_eda class
# eda = extensive_eda()
# eda.save_eda_html(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fe = clean()
    print(fe)
    print(f"\nColumns in dataset: {X.columns}\n")
    best_columns, _ = fe.correlation_multicollinearity(X)
    print(f"\nImportant columns after performing the mult-collinearity test using VIF approach are :{best_columns}\n")
    numerical_cols, category_cols = fe.find_numericategory_columns(X[best_columns])
    print("\nThe numeric columns are :", numerical_cols)
    print("\nThe categorical columns are :", category_cols)
    X_best = X[best_columns].copy()
    X_train, X_test, y_train, y_test = fe.split_train_test(X_best, y, test_size=0.008)
    # fe.preprocessor_fit(X_train, one_hot_encode_cols=category_cols, label_encode_cols=None)
    fe.preprocessor_fit(X_train, one_hot_encode_cols=None, label_encode_cols=category_cols)
    # Transform X_train
    X_train_transformed = fe.preprocessor_transform(X_train).to_numpy()
    # Create and tune the model
    ann = ANNClassifier(input_dim=X_train.shape[1], hidden_layers=[32, 16], output_dim=1, learning_rate=0.001,
                        classification_type='binary')
    ann.train(X_train_transformed, y_train, epochs=200, batch_size=32)
    ann.plot_loss_history()
    # Hyperparameter tuning
    # tuner = ann.tune(X_train_transformed, y_train, max_trials=10, executions_per_trial=1, directory=ticker)
    # print(f'Best hyperparameters: {tuner.get_best_hyperparameters(num_trials=1)[0].values}')

    # Assuming model is your trained ANN model
    # ann.model.save('ann_model.h5')  # Save the entire model as a .h5 file
    # # To load the model later
    #  model = load_model('ann_model.h5')
    # model.save('ann_model.keras')
    # # Load the model
    # model = load_model('ann_model.keras')

    # # Retrain the model with the best hyperparameters on the entire training set
    # #ann.build_model()  # Rebuild the model with the best hyperparameters
    # #ann.model.fit(X_train_transformed, y_train, epochs=100, batch_size=32, verbose=1)
    # # Evaluate the model with the best parameters on the test set
    X_test_transformed = fe.preprocessor_transform(X_test).to_numpy()
    test_loss = ann.evaluate(X_test_transformed, y_test)
    print(f'Test Loss with best hyperparameters: {test_loss}')
    ann.model_summary()
    y_pred = ann.predict(X_test_transformed)
    print(y_pred)
    y_pred = y_pred > 0.5
    print("The accuracy score is :", accuracy_score(y_pred, y_test))
    cm = confusion_matrix(y_pred, y_test)
    print()
    print(cm)
    fe.plot_confusion_matrix(y_pred, y_test)
    #
    # # # Final outputs
    X_best_transformed = fe.preprocessor_transform(X_best).to_numpy()
    df_new = df.copy()
    df_new['Predicted_Signal'] = ann.predict(X_best_transformed)
    # Calculate daily returns
    df_new['Return'] = df.Close.pct_change()
    # Calculate strategy returns
    df_new['Strategy_Return'] = df_new.Return * df_new.Predicted_Signal.shift(1)
    # Calculate Cumulative returns
    df_new['Cum_Ret'] = df_new['Return'].cumsum()
    # Plot Strategy Cumulative returns
    df_new['Cum_Strategy'] = df_new['Strategy_Return'].cumsum()
    # Create a new figure
    plt.figure(figsize=(12, 6))
    # Plot cumulative returns
    plt.plot(df_new['Cum_Ret'], color='red', label='Cumulative Return')
    # Plot cumulative strategy returns
    plt.plot(df_new['Cum_Strategy'], color='blue', label='Cumulative Strategy')
    # Add a title to the plot
    plt.title('Cumulative Returns and Strategy Returns')
    # Add a label to the x-axis
    plt.xlabel('Time')
    # # Add a label
    # to the y-axis
    plt.ylabel('Cumulative Return')
    # Add a legend to the plot
    plt.legend()
    # Show the plot
    plt.savefig(f'Cumulative Returns and Strategy Returns_{ticker}.png')

    # Predict the next-day
    X_nd = asset.generate_next_day_data(X_test.tail(1))
    X_combined = pd.concat([X_test, X_nd], axis=0)
    X_combined_transformed = fe.preprocessor_transform(X_combined).to_numpy()
    y_combined_pred = ann.predict(X_combined_transformed)
    print(y_combined_pred)
    # Interpret the prediction
    if y_combined_pred[-1] > 0.5:
        print("The model predicts that the stock price will go up tomorrow.")
    else:
        print("The model predicts that the stock price will go down tomorrow.")

    # Print prediction probabilities
    print(f"Probability of going up: {y_combined_pred[-1][0]:.2f}")
