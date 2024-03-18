def deep_calibration(
    tf_sess,
    nn,
    K_T,
    market_quotes,
    model_name="heston",
    lambd_init=0.1,
    beta0=0.25,
    beta1=0.75,
    max_iter=1000,
    tol=1e-8,
):
    """Combines LM algorithm with a NN regressor to calibrating model parameters."""
    # initialize model parameters
    params, param_names = model_parameters_initializer(model_name)

    # initalize learning step
    lambd = lambd_init

    n_samples = K_T.shape[0]
    n_params = len(params)
    I = np.eye(n_params)
    Q = market_quotes.reshape((-1, 1))  # shape: [n_samples, 1]
    K_T_values = K_T.values

    iter_count = 0

    # history to store some useful information during training
    history = {
        "delta_params": {k: [] for k in param_names},
        "R": [],
        "lambda": [],
        "c_mu": [],
    }

    # build a input dataframe by combining K_T and model parameters
    input_data = np.insert(
        K_T_values, [2] * n_params, params, axis=1
    )  # shape: [n_samples, n_params+2]
    iv_nn, J = predict_label_jac(tf_sess, nn, input_data)
    R = iv_nn - Q  # shape: [n_samples, 1]
    J = J[:, 2:]  # shape: [n_samples, n_params], excluding K and T
    delta_params = (
        np.linalg.pinv(J.T.dot(J) + lambd * I).dot(J.T.dot(R)).flatten()
    )  # vector size: [n_params,] ··· Cholesky better
    history["R"].append(np.linalg.norm(R))
    history["lambda"].append(lambd)
    for param_idx, param_name in enumerate(param_names):
        history["delta_params"][param_name].append(delta_params[param_idx])

    while iter_count < max_iter and np.linalg.norm(delta_params) > tol:
        if iter_count % 50 == 0:
            logging.info("{}/{} iteration".format(iter_count + 1, max_iter))
        params_new = params - delta_params
        input_data_new = np.insert(K_T_values, [2] * n_params, params_new, axis=1)
        iv_nn_new, J_new = predict_label_jac(tf_sess, nn, input_data_new)
        R_new = iv_nn_new - Q
        J_new = J_new[:, 2:]
        R_norm = np.linalg.norm(R)
        c_mu = (R_norm - np.linalg.norm(R_new)) / (
            R_norm - np.linalg.norm(R - J.dot(delta_params))
        )

        history["c_mu"].append(c_mu)

        if c_mu <= beta0:
            # reject delta_params
            lambd *= 2  # too slow, use greater lambd
        else:
            params = params_new
            R = R_new
            J = J_new
        if c_mu >= beta1:
            lambd /= 2.0

        delta_params = (
            np.linalg.pinv(J.T.dot(J) + lambd * I).dot(J.T.dot(R)).flatten()
        )  # vector size: [n_params, ]
        iter_count += 1

        history["R"].append(np.linalg.norm(R))
        history["lambda"].append(lambd)
        for param_idx, param_name in enumerate(param_names):
            history["delta_params"][param_name].append(delta_params[param_idx])
    if iter_count < max_iter:
        logging.info("Leave iterations after {} iters".format(iter_count))

    return dict(zip(param_names, params)), history


params, history = deep_calibration(
    sess,
    model,
    K_T_input,
    market_quotes,
    model_name="rBergomi",
    lambd_init=0.01,
    beta0=0.25,
    beta1=0.75,
    max_iter=1000,
)

market_quotes = np.array(
    [
        heston_pricer(
            lambd,
            vbar,
            eta,
            rho,
            v0,
            r,
            q,
            K_T_origin.iloc[i, 1],
            S0,
            K_T_origin.iloc[i, 0],
        )[1]
        for i in range(K_T.shape[0])
    ]
)
market_quotes = market_quotes.reshape((-1, 1))
