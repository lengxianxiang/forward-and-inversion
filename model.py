# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:28:25 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:38:21 2020

@author: user
"""
    import numpy as np
    from wavelets import ricker
    import time
    from gen_data import gen_data
    seam_model_path="周老师4000.txt"
    nx=200
    nz=200
    model= (np.fromfile(seam_model_path, sep=" ",dtype=np.float32)
                  .reshape([nx, nz]).T)

    #dx = 100   
    dx=20
    dt = 0.001
    num_sources = 1#90
    num_train_shots =10#80
    dsrc = 1
    num_receivers = 1#90
    drec = 1
    nt = 2// dt # 12 seconds
    sources = ricker(20, nt, dt, 1.0).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    sources_x = np.zeros([num_sources, 1, 3], np.int)
    sources_x[:, 0, 0] = 0
    sources_x[:, 0, 1] =100#z轴
    sources_x[:, 0, 2] = 100#x轴
    #sources_x[:, 0, 1] = np.arange(0, num_sources*dsrc, dsrc)
    #sources_x[:, 0, 2] = np.arange(0, num_sources*dsrc, dsrc)
    receivers_x = np.zeros([1, num_receivers, 3], np.int)
    receivers_x[0, :, 0] = 0
    receivers_x[0, :, 1] = 80
    receivers_x[0, :, 2] = 81
    #receivers_x[0, :, 1] = np.arange(0, num_receivers*drec, drec)
    #receivers_x[0, :, 2] = np.arange(0, num_receivers*drec, drec)
    #receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    #propagator = forward2d

    # The initial guess model will start at 1490 m/s at the top (water speed)
    # and increase by 0.5m/s for each meter in depth
    #v0 = 1490
    #dvdz = 0.5
    #model_init = (model
    #              + ((np.random.random(nx))*2).astype(np.float32))
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    start=time.perf_counter()
    out2=forward2d(model, sources, sources_x,
              dx, dt, pml_width=None, pad_width=None,
              profile=None)
    end=time.perf_counter()
    print(end-start)
    f1=sess.run(out2)[:,0,:,:]
    #receivers=a[:,receivers_x[0, :, 1],receivers_x[0, :, 2]]
   # receivers=tf.reshape(receivers,[int(nt),1,1])
