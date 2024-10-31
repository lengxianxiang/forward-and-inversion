import numpy as np
from wavelets import ricker
from PIL import Image
#from forward2d import forward2d

def setup_seam(seam_model_path):
    """Prepare the SEAM models and dataset metadata.

    Args:
        seam_model_path: The path to the file containing the true SEAM model

    Returns:
        A dictionary containing SEAM-related arrays and properties
    """
    #nx = 90
    #nz = 58
    #nx=756
    #nz=200
    nx=101
    nz=61
    model_true = (np.fromfile("sig", sep=" ",dtype=np.float32)
                  .reshape([nz, nx]))
   # model_true=model_true[:,np.arange(101)]
    #model_true=model_true[np.arange(61),:]
    model_init = (np.fromfile("isig", sep=" ",dtype=np.float32)
                .reshape([nz, nx]))
    #model_init=model_init[:,np.arange(101)]
    #model_init=model_init[np.arange(61),:]
    
    #dx = 100
    dx=8
    dt = 0.001
    num_sources = 101#90
    num_train_shots =80#80
    dsrc = 1
    num_receivers = 101#90
    drec = 1
    nt = int(2 // dt) # 12 seconds
    sources = ricker(20, nt, dt, 1.0).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    sources_x = np.zeros([num_sources, 1, 2], np.int)
    sources_x[:, 0, 0] =30
    sources_x[:, 0, 1]=np.arange(101)
    #sources_x[:, 0, 1] = np.arange(0, num_sources*dsrc, dsrc)
    #sources_x[:, 0, 2] = np.arange(0, num_sources*dsrc, dsrc)
    receivers_x = np.zeros([1, num_receivers, 2], np.int)
    receivers_x[0, :, 0] =0
    #receivers_x[0, 64:64*2, 0] =30
    #receivers_x[0, 64*2:, 0] = 58
    receivers_x[0,:,1]=np.arange(101)
    #receivers_x[0,64:64*2,1]=2*np.arange(64)
    #receivers_x[0,64*2:,1]=2*np.arange(64)
    #receivers_x[0, :, 1] = np.arange(0, num_receivers*drec, drec)
    #receivers_x[0, :, 2] = np.arange(0, num_receivers*drec, drec)
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    #propagator = forward2d

  
    #model_init=model_true*1.1
    #nums=[]
    #for i in range(nx*nz):
     #   temp=random.gauss(0,200)
     #   nums.append(temp)
    #nums=np.array(nums,dtype=np.float32)
    #model_init=model_true+nums.reshape((nz,nx))
    #model_init=model_true+((-2+4*np.random.random(nx*nz)*550).astype(np.float32)).reshape((nz,nx))
    #model_init = np.arange(v0, v0+nz*dx*dvdz, dx*dvdz).astype(np.float32)
    #model_init = np.tile(model_init.reshape([-1, 1]), [1, nx])

    return {'model_true': model_true,
            'model_init': model_init,
            'dx': dx,
            'dt': dt,
            'num_train_shots': num_train_shots,
            'sources': sources,
            'sources_x': sources_x,
            'receivers_x': receivers_x}
            