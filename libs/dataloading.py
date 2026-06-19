import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split


# --- whole data set ---
DATA_PATH_TEMPLATE = "/work/bd1179/b309252/QML2/QML2_{}_cg200_10z.txt"
SERIES = [f"{s}{i}" for s in "abc" for i in range(1, 7)]
SERIES_GEN = [f"{s}{i}" for s in "d" for i in range(1, 7)]
MAX_SAMPLES = 610800
TEST_SIZE = 122100


def load_all_data(series_labels):
    df_all = pd.DataFrame()
    for label in series_labels:
        df = pd.read_csv(DATA_PATH_TEMPLATE.format(label)).dropna()
        df_all = pd.concat([df_all, df])
    return df_all

    
## preprocessing to scale the inputs to the interval [-pi/2,pi/2]
def preprocess_features(X, X_ref, tanh_scale = 0.35):
    return pd.DataFrame({
            'p': 0.5*np.pi*np.tanh(X.p / X_ref.p.std() * 1.8*tanh_scale),
            'theta': 0.5*np.pi*np.tanh((X.theta - X_ref.theta.mean()) / X_ref.theta.std() * 0.8*tanh_scale),
            'zrel': (X.zrel_u - 0.5) * np.pi,
            'u_cg': 0.5*np.pi*np.tanh((X.u_cg - X_ref.u_cg.mean()) / X_ref.u_cg.std()*tanh_scale),
            'v_cg': 0.5*np.pi*np.tanh((X.v_cg - X_ref.v_cg.mean()) / X_ref.v_cg.std()*tanh_scale),
            'w_cg':0.5*np.pi*np.tanh(X.w_cg / X_ref.w_cg.std()*tanh_scale)
        })

## preprocessing to scale the target to the interval [-1,1] corresponding to possible QNN expectation values  
def preprocess_target(y, y_ref = 'all', tanh_scale = .65):
    if (type(y_ref) == str ) and (y_ref == 'all'):
        df_all = load_all_data(SERIES)
        df_all = df_all[(df_all.zrel_u > 0.2) & (df_all.zrel_u < 0.8)].iloc[:MAX_SAMPLES]
        y_ref = df_all[['theta_w_SGS']].values
    y_norm = (y - y_ref.mean()) / y_ref.std() * tanh_scale
    return np.tanh(y_norm)

## undo the preprocessing of the target to get the physical values
def unpreprocess_target_unsafe(Y, y_ref = 'all', tanh_scale = .65):
    if (type(y_ref) == str ) and (y_ref == 'all'):
        df_all = load_all_data(SERIES)
        df_all = df_all[(df_all.zrel_u > 0.2) & (df_all.zrel_u < 0.8)].iloc[:MAX_SAMPLES]
        y_ref = df_all[['theta_w_SGS']].values
    return np.arctanh(Y) *y_ref.std() / tanh_scale + y_ref.mean()

## safe version of the back conversion
def unpreprocess_target(Z, y_ref = 'all',  tanh_scale = .65):
    if (type(y_ref) == str ) and (y_ref == 'all'):
        df_all = load_all_data(SERIES)
        df_all = df_all[(df_all.zrel_u > 0.2) & (df_all.zrel_u < 0.8)].iloc[:MAX_SAMPLES]
        y_ref = df_all[['theta_w_SGS']].values
        
    invalid_mask = (Z <= -1) | (Z >= 1)

    if np.any(invalid_mask):
        print("Invalid values encountered in arctanh input!")
        print("Indices:", np.where(invalid_mask))
        print("Offending values (Z):", Z[invalid_mask])
        print("Corresponding original Y values:", Y[invalid_mask])
        print(f"Min Z: {Z.min()}, Max Z: {Z.max()}")

    # Clip slightly to avoid NaNs in arctanh
    Z = np.clip(Z, -0.999999, 0.999999)

    return np.arctanh(Z) * y_ref.std()/ tanh_scale + y_ref.mean()

## data loading for the training   
def load_data(seed, n_samples = 'all'):
    
    df_all = load_all_data(SERIES)
    df_all = df_all[(df_all.zrel_u > 0.2) & (df_all.zrel_u < 0.8)].iloc[:MAX_SAMPLES]
    
    if n_samples == 'all':
        test_size = TEST_SIZE
        df_sel = df_all
    else:
        test_size = round(0.002*n_samples) *100
        df_sel = df_all.sample(n = n_samples, random_state = seed)
        
    X = preprocess_features(df_sel[['p','theta','zrel_u','u_cg','v_cg','w_cg', 'theta_v','theta_u','theta_w']], df_all[['p','theta','zrel_u','u_cg','v_cg','w_cg', 'theta_v','theta_u','theta_w']])

    y = preprocess_target(df_sel[['theta_w_SGS']], df_all[['theta_w_SGS']])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    jnp_train_inputs = jnp.array(X_train)
    jnp_val_inputs = jnp.array(X_val)
    jnp_train_outputs = jnp.array(y_train).flatten()
    jnp_val_outputs = jnp.array(y_val).flatten()

    return jnp_train_inputs, jnp_val_inputs, jnp_train_outputs, jnp_val_outputs

## data loading for the generalisation tests
def load_gen_data(seed, n_samples = 'all', series_to_load = SERIES_GEN):
    
    df_all = load_all_data(SERIES)
    df_all = df_all[(df_all.zrel_u > 0.2) & (df_all.zrel_u < 0.8)].iloc[:MAX_SAMPLES]

    df_gen = load_all_data(series_to_load)
    df_gen = df_gen[(df_gen.zrel_u > 0.2) & (df_gen.zrel_u < 0.8)]
    
    if n_samples == 'all':
        test_size = TEST_SIZE
        df_sel = df_gen
    else:
        test_size = round(0.002*n_samples) *100
        df_sel = df_gen.sample(n = n_samples, random_state = seed)
        
    X = preprocess_features(df_sel[['p','theta','zrel_u','u_cg','v_cg','w_cg']], df_all[['p','theta','zrel_u','u_cg','v_cg','w_cg']])
    
    y = preprocess_target(df_sel[['theta_w_SGS']], df_all[['theta_w_SGS']])
    
    jnp_val_inputs = jnp.array(X)
    jnp_val_outputs = jnp.array(y).flatten()

    return jnp_val_inputs, jnp_val_outputs