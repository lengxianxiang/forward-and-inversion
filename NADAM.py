#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 02:54:21 2022

@author: user
"""


import sys
import csv
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
from fwi import (Fwi, shuffle_shots, extract_datasets, _get_dev_loss)
from setup_seam import setup_seam

def make_adam_vs_lbfgs_figure():
    """Create and plot data comparing the Adam and L-BFGS-B optimizers.
    """
    np.random.seed(0)

    model_true, model_init, dx, dt, train_dataset, dev_dataset, propagator = \
            _prepare_datasets()

    model_inv_adam, model_inv_sgd, model_inv_lbfgs, \
            adam_evals, dev_loss_adam, \
            sgd_evals, dev_loss_sgd, \
            lbfgs_evals, dev_loss_lbfgs = _make_figure_data(model_init,
                                                            dx, dt,
                                                            train_dataset,
                                                            dev_dataset,
                                                            propagator)

    _save_figure_data(model_inv_adam, model_inv_sgd, model_inv_lbfgs,
                      adam_evals, dev_loss_adam,
                      sgd_evals, dev_loss_sgd,
                      lbfgs_evals, dev_loss_lbfgs)

    _make_plots(model_inv_adam, model_inv_sgd, model_inv_lbfgs,
                adam_evals, dev_loss_adam,
                sgd_evals, dev_loss_sgd,
                lbfgs_evals, dev_loss_lbfgs,
                model_true, model_init, dx)


def _prepare_datasets():
    """Load the initial model, datasets, and other relevant values."""
    seam_model_path = sys.argv[1]
    seam = setup_seam(seam_model_path)
    model_true = seam['model_true']
    model_init = seam['model_init']
    dx = seam['dx']
    dt = seam['dt']
    sources = seam['sources']
    sources_x = seam['sources_x']
    receivers_x = seam['receivers_x']
    propagator = seam['propagator']
    num_train_shots = seam['num_train_shots']

    seam_data_path = sys.argv[2]
    receivers = np.load(seam_data_path)
    #a=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\单层接收.txt", sep=" ",dtype=np.float32)
            #      .reshape([999, 50,1]))
    dataset = sources, sources_x, receivers, receivers_x

    dataset = shuffle_shots(dataset)
    train_dataset, dev_dataset = extract_datasets(dataset, num_train_shots)

    return (model_true, model_init, dx, dt, train_dataset, dev_dataset,
            propagator)


def _make_figure_data(model_init, dx, dt, train_dataset, dev_dataset,
                      propagator):
    """Run optimization using the Adam optimizer (with the best hyperparameters
    found from make_hyperparameter_selection_figure) and L-BFGS-B (from
    Scipy).

    Returns:
        model_inv_adam: The final Adam model
        model_inv_sgd: The final SGD model
        model_inv_lbfgs: The final L-BFGS-B model
        adam_evals: A list containing the number of shots that have been
                    evaluated (forward and backward propagated to calculate
                    gradient) corresponding to dev_loss_adam
        dev_loss_adam: A list containing the cost function value after the
                       number of shot evaluations in the corresponding list
                       entry in adam_evals, when using Adam
        sgd_evals: Same as adam_evals, but for SGD
        dev_loss_sgd: Same as dev_loss_adam, but for SGD
        lbfgs_evals: Same as adam_evals, but for L-BFGS-B
        dev_loss_lbfgs: Same as dev_loss_adam, but for L-BFGS-B
    """
    
    seam_model_path=""
    seam = setup_seam(seam_model_path)
    
    model_init = seam['model_init']
    dx = seam['dx']
    dt = seam['dt']
    sources = seam['sources']
    sources_x = seam['sources_x']
    receivers_x = seam['receivers_x']
    propagator = forward2d
    propagator1 = forward2d_lap
    propagator2 = forward2d_tt
    num_train_shots = seam['num_train_shots']
    model=seam['model_true']
    receivers=gen_data(model, dx, dt, sources, sources_x, receivers_x, propagator,
             batch_size=1)
    dataset = sources, sources_x, receivers, receivers_x
    dataset = shuffle_shots(dataset)
    train_dataset, dev_dataset = extract_datasets(dataset, num_train_shots)
   
    import time
    start=time.perf_counter()
    num_epochs =4# only use 5 passes through the whole dataset
    num_train_shots = train_dataset[0].shape[1]

    # Adam
    #optimizer = tf.train.AdamOptimizer(15)
    #optimizer = tf.train.AdagradOptimizer(15)
    optimizer = tf.contrib.opt.NadamOptimizer(30)
    batch_size =3
    num_batches_in_one_epoch = num_train_shots // batch_size
    num_steps = num_epochs * num_batches_in_one_epoch
    fwi = Fwi(model,model_init, dx, dt, train_dataset, dev_dataset, propagator,propagator1,propagator2,optimizer=optimizer,batch_size=batch_size)
    model_inv_nadam7, loss_adam7= fwi.train(num_steps, 1)
    end=time.perf_counter()
    print(end-start)

    # Stochastic Gradient Descent
    tf.reset_default_graph()
    optimizer = tf.train.GradientDescentOptimizer(50)
    batch_size = 5
    num_batches_in_one_epoch = num_train_shots // batch_size
    num_steps = num_epochs * num_batches_in_one_epoch
    fwi = Fwi(model_inv_nadam9, model,dx, dt, train_dataset, dev_dataset, propagator,
              optimizer=optimizer,
              batch_size=batch_size)
    model_inv_sgd, loss_sgd = fwi.train(num_steps, 1)

    # L-BFGS-B
    # I limit the maximum number of epochs to 'num_epochs + 10' (+10 to account
    # for the equivalent of 10 passes through the data that I used to
    # choose the Adam hyperparameters). Note, however, that it actually
    # evaluates the function more than "maxfun" times, presumably as it
    # evaluates more than once per iteration and only checks this limit
    # before starting a new iteration.
    tf.reset_default_graph()
    batch_size = 5 # Only affects computational performance, not result
    loss_file = 'lbfgs_loss.txt' # temporary file to write loss values to
    fwi = Fwi(model_init, dx, dt, train_dataset, dev_dataset, propagator,
              batch_size=batch_size, save_gradient=True)
    res = fwi.train_lbfgs({'maxfun': num_epochs , 'gtol': 1e-20},
                          loss_file=loss_file)
    model_inv_lbfgs = res.x.reshape(model_init.shape)

    # Open loss_file and read the loss values into an array
    loss_lbfgs = []
    with open(loss_file, newline='') as csvfile:
        lossreader = csv.reader(csvfile)
        for row in lossreader:
            loss_lbfgs.append(row)

    # Need to calculate the L-BFGS-B dev loss one more time to get the value
    # for the final model (unlike Adam, the dev loss for L-BFGS-B is
    # calculated before the model is updated)
    dev_loss = _get_dev_loss(model_inv_lbfgs.shape, dev_dataset,
                             fwi.batch_placeholders,
                             fwi.loss, fwi.sess, model=fwi.model,
                             test_model=model_inv_lbfgs)
    loss_lbfgs.append([0, dev_loss])
    loss_lbfgs = np.array(loss_lbfgs, np.float32)

    # We use the [:, 1] entries, as [:, 0] correspond to training losses, but
    # we want development dataset losses
    dev_loss_adam15= np.array(loss_adam15)[1][:, 1]
    dev_loss_sgd = np.array(loss_sgd)[1][:, 1]
    # Since dev_loss for Adam/SGD is calculated after updating the model, I copy
    # the first value (for 0 shot evaluation) from the L-BFGS-B results.
    # This is also the reason for the "+ 1" in the line after.
    #dev_loss_adam = np.insert(dev_loss_adam, 0, loss_lbfgs[0, 1])
    #dev_loss_sgd = np.insert(dev_loss_sgd, 0, loss_lbfgs[0, 1])
    adam_evals = batch_size * np.arange(num_steps )
    sgd_evals = batch_size * np.arange(num_steps )

    dev_loss_lbfgs = np.array(loss_lbfgs)[:, 1].astype(np.float)
    lbfgs_evals = num_train_shots * np.arange(len(dev_loss_lbfgs))

    return (model_inv_adam, model_inv_sgd, model_inv_lbfgs,
            adam_evals, dev_loss_adam,
            sgd_evals, dev_loss_sgd,
            lbfgs_evals, dev_loss_lbfgs)


def _save_figure_data(model_inv_adam, model_inv_sgd, model_inv_lbfgs,
                      adam_evals, dev_loss_adam,
                      sgd_evals, dev_loss_sgd,
                      lbfgs_evals, dev_loss_lbfgs):
    """Save the data that is needed to generate the plots so that they can be
    recreated without running the computations again, if necessary.
    """
    np.save('adam_vs_lbfgs_model_inv_adam', model_inv_adam)
    np.save('adam_vs_lbfgs_model_inv_sgd', model_inv_sgd)
    np.save('adam_vs_lbfgs_model_inv_lbfgs', model_inv_lbfgs)
    np.save('adam_vs_lbfgs_adam_evals', adam_evals)
    np.save('adam_vs_lbfgs_dev_loss_adam', dev_loss_adam)
    np.save('adam_vs_lbfgs_sgd_evals', sgd_evals)
    np.save('adam_vs_lbfgs_dev_loss_sgd', dev_loss_sgd)
    np.save('adam_vs_lbfgs_lbfgs_evals', lbfgs_evals)
    np.save('adam_vs_lbfgs_dev_loss_lbfgs', dev_loss_lbfgs)


def _make_plots(model_inv_adam, model_inv_sgd, model_inv_lbfgs,
                adam_evals, dev_loss_adam,
                sgd_evals, dev_loss_sgd,
                lbfgs_evals, dev_loss_lbfgs,
                model_true, model_init, dx):
    """Create plots of the cost function value vs. number of shot evaluations
    for Adam and L-BFGS-B, and of a comparison between the true and initial
    models and the final models produced by Adam and SGD, and save both
    as EPS files.
    """
    # Loss plot
    _, _ = plt.subplots(figsize=(5.4, 3.3))
    plt.style.use('grayscale')
    plt.rc({'font.size': 8})
    #plt.plot(lbfgs_evals, dev_loss_lbfgs, '.--', label='L-BFGS-B')
    plt.plot(sgd_evals, dev_loss_sgd, 'o:', label='SGD')
    plt.plot(adam_evals, dev_loss_adam, '.-', label='Adam')
    plt.xlabel('Number of shot evaluations')
    plt.ylabel('Loss')
    plt.legend(loc=7)
    plt.title('Adam converges faster')
    plt.tight_layout(pad=0)
    plt.savefig('adam_vs_lbfgs_loss.jpg')
    
    model_true=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_true_m2.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    model_init=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_init_m2.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    model_inv_nadam=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_m2.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    
    model_true=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_m2_onad.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    model_init=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_m2_onsprk3.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    model_inv_nadam=(np.fromfile("D:\研究生\研究生task ing\深度学习与地球物理\model_m2_nsprk8.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    m1=(np.fromfile("a6.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))
    m2=(np.fromfile("a11.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))   
    m3=(np.fromfile("a12.txt", sep=" ",dtype=np.float32)
                  .reshape([15, 51]))   
       
    # Model plots
    fig = plt.figure(figsize=(8.1, 4.5))
    #plt.style.use('grayscale')
    plt.style.use('classic')
    plt.rc({'font.size': 8})
    plt.rcParams['font.size'] =7
    extent = [0, model.shape[1]*dx/1e3, model.shape[0]*dx/1e3, 0]
    aspect = 'equal'
    vmin =1400/1e3
    vmax = 4600/1e3
    gs1 = gridspec.GridSpec(2,3, width_ratios=[1,1,1],
                            height_ratios=[1, 0.045])
    ax = []
    ax.append(plt.subplot(gs1[0]))
    ax.append(plt.subplot(gs1[1]))
    ax.append(plt.subplot(gs1[2]))
    #ax.append(plt.subplot(gs1[3]))
    im = ax[0].imshow(model/1e3, extent=extent, aspect=aspect,\
                      vmin=vmin, vmax=vmax)
    ax[0].set_title('True')
    ax[0].set_xlabel('x (km)')
    ax[0].set_ylabel('Depth (km)')
    #plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].imshow(model_init/1e3, extent=extent, aspect=aspect,
                 vmin=vmin, vmax=vmax)
    ax[1].set_title('Initial')
    ax[1].set_xlabel('x (km)')
    ax[1].set_ylabel('Depth (km)')
    #plt.setp(ax[1].get_xticklabels(), visible=False)
    #plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[2].imshow(model_inv_nadam7/1e3, extent=extent, aspect=aspect,
     vmin=vmin, vmax=vmax)
    ax[2].set_title('TMSOS+RNN')
    ax[2].set_xlabel('x (km)')
    ax[2].set_ylabel('Depth (km)')
   # ax[3].imshow(model_inv_nadam1/1e3, extent=extent, aspect=aspect,\
   #              vmin=vmin, vmax=vmax)
   # ax[3].set_title('mse_3_n_3')
   # plt.setp(ax[3].get_yticklabels(), visible=False)
   # ax[3].set_xlabel('x (km)')
    cax = plt.subplot(gs1[-3:])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Wave speed (km/s)')
    plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 0.98])
    plt.savefig('m80.jpg')


if __name__ == '__main__':
    make_adam_vs_lbfgs_figure()