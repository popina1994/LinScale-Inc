import numpy as np
import itertools
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import time

# Start time


def cartesian_product_matrices(A, B):
    A_rows = A.shape[0]
    B_rows = B.shape[0]

    # Cartesian product of row indices
    cartesian_indices = np.array(list(itertools.product(range(A_rows), range(B_rows))))

    # Use indices to stack corresponding rows from A and B
    result = np.hstack((A[cartesian_indices[:, 0]], B[cartesian_indices[:, 1]]))

    return result


def xgboost_try_out(data):
    # Split features (X) and target (y)
    X = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Last column

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.reshape(X_train.shape[0], 1)
    y_test = y_test.reshape(X_test.shape[0], 1)
    # Create DMatrix for XGBoost
    X_trainGPU = xgb.DMatrix(X_train, label=y_train)
    X_testGPU = xgb.DMatrix(X_test, label=y_test)
    start_time = time.time()
    # Initialize XGBoost model


    # Set XGBoost parameters for GPU
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda'
    }

    # Train the XGBoost model
    model = xgb.train(params, X_trainGPU, num_boost_round=100,)

    # Evaluate the model
    predictions = model.predict(X_testGPU)
    # model = xgb.XGBRegressor(device='cuda', objective='reg:squarederror', n_estimators=100)
    # # Train the model
    # model.fit(X_train, y_train)

    # # Predict on test set
    # y_pred = model.predict(X_test)
    end_time = time.time()

    # print(y_test, y_pred)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"XGBoost Mean Squared Error: {mse:.4f}")
    # Calculate the elapsed time in seconds
    elapsed_time_seconds = end_time - start_time

    # Convert seconds to milliseconds
    elapsed_time_milliseconds = elapsed_time_seconds * 1000

    print(f"Elapsed time: {elapsed_time_milliseconds} milliseconds")

def lin_reg_try_out(data):
    # Example data
    X = data[:, :-1]
    y = data[:, -1]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    start_time = time.time()
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    end_time = time.time()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    # print("OUTPUT", y_test)
    print(f"Linear regression Mean Squared Error: {mse}")

    elapsed_time_seconds = end_time - start_time

    # Convert seconds to milliseconds
    elapsed_time_milliseconds = elapsed_time_seconds * 1000

    print(f"Elapsed time: {elapsed_time_milliseconds} milliseconds")


def lin_reg_qr_try_out(data):
    # Example data
    X = data[:, :-1]
    y = data[:, -1]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, shuffle=False)

    start_time = time.time()
    # # Add a bias column to X (for the intercept)
    # X_btrain = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Adding a column of ones for the intercept
    # X_btest = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Adding a column of ones for the intercept
    Xlen = int(X.shape[0])
    X_btrain = X_train
    X_btest = X_test

    # Perform QR decomposition
    # Q, R = np.linalg.qr(X_btrain)
    # R = np.loadtxt('200x2,1000x2LinScale', delimiter=',', dtype=float)
    R = np.loadtxt('200x2,1000x2CUDA', delimiter=',', dtype=float)
    # R = np.linalg.qr(X_btrain, mode='r')
    np.savetxt("R.csv", R, delimiter=",")

    # Compute beta: R^{-1} * Q.T * y
    # beta = np.linalg.inv(R).dot(Q.T).dot(y_train)

    # Compute beta using the Normal Equation
    # beta = np.linalg.inv(R.T.dot(R)).dot(X_btrain.T).dot(y_train)

    beta = np.linalg.inv(R.T.dot(R)).dot(X_btrain.T).dot(y_train)

    # print(f"Model coefficients (beta): {beta}")

    # Make predictions
    y_pred = X_btest.dot(beta)
    end_time = time.time()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    # print("OUTPUT", y_test)
    print(f"Linear regression LinScale QR Mean Squared Error: {mse}")

    elapsed_time_seconds = end_time - start_time

    # Convert seconds to milliseconds
    elapsed_time_milliseconds = elapsed_time_seconds * 1000

    print(f"Elapsed time: {elapsed_time_milliseconds} milliseconds")

if __name__ == "__main__":
    # Example matrices
    np.random.seed(1)
    A =  np.random.randint(1, 10, (1000, 2))  # 3x4 matrix with random integers from 1 to 9
    B = np.random.randint(1, 10, (1000, 2))  # Shape (2, 2)
    np.savetxt("A.csv", A[0:200, :], delimiter=",")
    np.savetxt("B.csv", B, delimiter=",")
    # print(A)
    # print(B)
    result = cartesian_product_matrices(A, B)
    random_col = np.random.randn(result.shape[0]) / 100
    print(random_col.shape, result[:, 0].shape)
    print(random_col)
    out = result[:, 0] + 2 * result[:, 1] + 3 * result[:, 2] + 4 * result[:, 3] +  random_col
    out = out.reshape((result.shape[0], 1))
    data = np.concatenate((result, out), axis=1)
    # print(data)
    xgboost_try_out(data)
    # lin_reg_try_out(data)
    lin_reg_qr_try_out(data)

