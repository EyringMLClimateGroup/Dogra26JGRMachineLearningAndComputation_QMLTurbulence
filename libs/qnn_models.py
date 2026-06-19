import csv
import os
import sys
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import pennylane as qml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from functools import reduce
import json
import re

import libs.circuits
import libs.dataloading

#### generating and training QML models based on PQC from circuits.py and a local or global observable from Z-measurements

# --- Model, Loss, Optimizer ---
def make_model(dev, circuit, local_observable):
    model_name = str(circuit.n_wires) + 'qubits_' + circuit.circuit_name + str(circuit.n_enc)+'enc_' + str(circuit.n_dec)+'dec_' +('' if local_observable else 'non') + 'local'
    no_gate_angles = circuit.n_var_angles
    no_upload_weights = circuit.n_encoding_angles if circuit.choose_freq else 0
    no_params = no_gate_angles  + no_upload_weights 

    if local_observable:
        observable_indices = list(range(circuit.n_wires))
        obs_return = lambda: [qml.expval(qml.PauliZ(i)) for i in observable_indices]
    else:
        full_obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(circuit.n_wires)])
        obs_return = lambda: [qml.expval(full_obs)]

    @qml.qnode(dev, interface="jax")
    def qnn_pqc(inputs, angles):
        circuit(inputs, angles,  dev.wires)
        return obs_return()

    @jax.jit
    def model(params, inputs):
        outputs = qnn_pqc(inputs,params)
        return jnp.squeeze(outputs[0])

    return model, no_params, model_name


def train_model(args_opt, model, params, train_inputs, train_targets, val_inputs, val_targets):
    opt, loss, no_epochs, batch_size = args_opt
    
    trainset_size = train_inputs.shape[0]
    valset_size = val_inputs.shape[0]
    no_features = train_inputs.shape[1]
    no_batches_train = int(trainset_size / batch_size)
    no_batches_val = int(valset_size / batch_size)

    ### compile the loss function
    @jax.jit
    def loss_fn(params, inputs, targets):
        return loss(params, inputs, targets, model=model)
    
    ### compile the update step for one batch
    @jax.jit
    def update_step_batch_jit(i, args):
        params, opt_state, in_train, out_train = args
        ins = jax.lax.dynamic_slice(in_train, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_train, [i*batch_size], [batch_size])
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ins, outs)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, in_train, out_train)

    ### compile function for calculating loss over one batch
    @jax.jit
    def calculate_dataset_loss_jit(i, args):
        loss_value, params, in_data, out_data = args
        ins = jax.lax.dynamic_slice(in_data, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_data, [i*batch_size], [batch_size])
        loss_batch = loss_fn(params, ins, outs)
        loss_value = loss_value + loss_batch
        return (loss_value, params, in_data, out_data)
    
    ### compile the update step for one epoch
    @jax.jit
    def update_step_epoch_jit(i, args):
        params, loss_hist, opt_state, in_train, out_train, in_val, out_val = args
        
        # shuffle training data and loop over batches
        inputs, outputs = shuffle(in_train, out_train)
        args_batch = (params, opt_state, inputs, outputs)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, no_batches_train, update_step_batch_jit, args_batch)
        
        # calculate train and validation losses at the end of each epoch
        train_loss = 0.0; args_train_loss = (train_loss, params, in_train, out_train)
        (train_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_train, calculate_dataset_loss_jit, args_train_loss)
        train_loss = train_loss / no_batches_train
        val_loss = 0.0; args_val_loss = (val_loss, params, in_val, out_val)
        (val_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_val, calculate_dataset_loss_jit, args_val_loss)
        val_loss = val_loss / no_batches_val

        # update loss history container
        train_hist = loss_hist['train_loss']
        train_hist = train_hist.at[i].set(train_loss)
        loss_hist['train_loss'] = train_hist
        val_hist = loss_hist['val_loss']
        val_hist = val_hist.at[i].set(val_loss)
        loss_hist['val_loss'] = val_hist

        # Print the loss 
        jax.debug.print("Epoch {ep}  ----  Train loss: {train_loss}  ----  Val. loss: {val_loss}", ep=(i+1), train_loss=train_loss, val_loss=val_loss)
        
        return (params, loss_hist, opt_state, in_train, out_train, in_val, out_val)
    
    ### compile the optimization loop
    @jax.jit
    def optimization_jit(params, train_data, val_data):
        in_train, out_train = train_data
        in_val, out_val = val_data

        # initialize optimizer and loss history container
        opt_state = opt.init(params)
        loss_history = dict()
        loss_history['train_loss'] = jnp.zeros(no_epochs)
        loss_history['val_loss'] = jnp.zeros(no_epochs)

        # run optimization loop
        args = (params, loss_history, opt_state, in_train, out_train, in_val, out_val)
        (params, loss_history, opt_state, _, _, _, _) = jax.lax.fori_loop(0, no_epochs, update_step_epoch_jit, args)
    
        return params, loss_history
    
    ### run the optimization
    train_data = (train_inputs, train_targets)
    val_data = (val_inputs, val_targets)
    opt_params, loss_history = optimization_jit(params, train_data, val_data)
    
    return opt_params, loss_history

# mean squared error loss function
def mse_loss(params, inputs, targets, model):
    predictions = model(params, inputs)
    loss = jnp.sum((targets - predictions) ** 2.0) / len(targets)
    return loss

    
# --- Training and evaluation ---
def train_and_evaluate(model, model_name, no_params, 
                       key, 
                       train_inputs, train_outputs, val_inputs, val_outputs,
                       save_dir,
                       initial_params = 'random_uniform',
                       return_optpars = False,
                       save_postfix = '',
                       loss = mse_loss,
                       plot_hist = True,
                       learning_rate = 0.002, n_epochs = 150, batch_size = 50):

    os.makedirs(save_dir, exist_ok=True)

    match = re.match(r"(\d+)qubits.*?(\d+)enc_(\d+)dec_([a-zA-Z]+)", model_name)
    if match:
        n_qubits, n_enc, n_dec, entanglement = match.groups()
        n_qubits, n_enc, n_dec, entanglement = int(n_qubits), int(n_enc), int(n_dec), int(entanglement == 'local')

    if (type(initial_params) == str) and (initial_params == 'random_uniform'):
        params = jax.random.uniform(key, shape=(no_params,), minval=-1.0, maxval=1.0)
        if 'freq' in model_name:
            params = params.at[:n_enc*n_qubits].set(params[:n_enc*n_qubits] * 0.01 + 1)
    else:
        params = initial_params
    initial_params_vector = params
    opt = optax.adam(learning_rate=learning_rate)
    

    @jax.jit
    def train_model_jit(params):
        args = (opt, loss, n_epochs, batch_size)
        return train_model(args, model, params, train_inputs, train_outputs, val_inputs, val_outputs)

    start = time.time()
    opt_params, history = train_model_jit(params)
    print(f"Training time: {time.time() - start:.2f} seconds")

    if plot_hist:
        plt.figure()
        plt.title(model_name)
        cols = ['firebrick','darkblue']
        i = 0
        for var in ['train_loss', 'val_loss']:
            plt.plot(history[var], label = var,color = cols[i])
            i+=1
        plt.legend()
        plt.show()
    
    os.makedirs(save_dir, exist_ok=True)
    
    y_train_hat = model(opt_params, train_inputs)
    y_val_hat = model(opt_params, val_inputs)

    y_train_hat_uz = libs.dataloading.unpreprocess_target(y_train_hat)
    y_val_hat_uz = libs.dataloading.unpreprocess_target(y_val_hat)
    
    y_train_uz = libs.dataloading.unpreprocess_target(train_outputs)
    y_val_uz = libs.dataloading.unpreprocess_target(val_outputs)

    metrics = {
        'r2_train_scaled': r2_score(train_outputs, y_train_hat),
        'r2_val_scaled': r2_score(val_outputs, y_val_hat),
        'mse_train': mean_squared_error(train_outputs, y_train_hat),
        'mse_val': mean_squared_error(val_outputs, y_val_hat),
        'r2_train': r2_score(y_train_uz, y_train_hat_uz),
        'r2_val': r2_score(y_val_uz, y_val_hat_uz)
    }

    with open(f"{save_dir}/QNN_r2.txt", mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([save_postfix, model_name, metrics['r2_train'], metrics['r2_val'], metrics['r2_train_scaled'], metrics['r2_val_scaled'],  metrics['mse_train'], metrics['mse_val']])


    # --- Save ---
    np.savez_compressed(f"{save_dir}/{model_name}_optparams{save_postfix}.npz", opt_params=opt_params)
    np.savez_compressed(f"{save_dir}/{model_name}_history{save_postfix}.npz", history=history)
    
    if return_optpars:
        return metrics, opt_params, history, initial_params_vector
    else:
        return metrics