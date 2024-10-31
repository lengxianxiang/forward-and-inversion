import numpy as np
import tensorflow as tf
from gen_data import gen_data

class TimeStepCell(tf.contrib.rnn.RNNCell):
    """One forward modeling step of scalar wave equation with PML.

    Args:
        model_padded2_dt2: Tensor containing squared wave speed times squared
            time step size
        dt: Float specifying time step size
        sigmaz: 1D Tensor that is only non-zero in z direction PML regions
        sigmax: 1D Tensor that is only non-zero in x direction PML regions
        first_z_deriv: Function to calculate the first derivative of the input
                       2D Tensor in the z direction
        first_x_deriv: Function to calculate the first derivative of the input
                       2D Tensor in the x direction
        laplacian: Function to calculate the Laplacian of the input
                   2D Tensor
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source

    """

    def __init__(self, model_padded2_dt2, dt, sigmaz, sigmax,
                 first_z_deriv3, first_x_deriv3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15,
                 sources_x):
        super(TimeStepCell, self).__init__()
        self.model_padded2_dt2 = model_padded2_dt2
        self.dt = dt
        self.c1=0.2633
        self.c2=-0.0486
        self.c3=0.4126
        self.c4=0.3727
        self.d1=-0.3598
        self.d2=0.7514
        self.d3=0.4696
        self.d4=0.1388
        self.sigmaz = sigmaz
        self.sigmax = sigmax
        self.ax=1+0.1*sigmax
        self.az=1+0.1*sigmaz
        self.sigma_sum = sigmaz + sigmax
        self.a_sum = self.ax+self.az
        self.sigma_prod_dt2 = (sigmaz * sigmax)
        self.factor = 1 / (1 + dt * self.sigma_sum / 2)
        self.first_z_deriv3 = first_z_deriv3
        self.first_x_deriv3 = first_x_deriv3
        #self.laplacian = laplacian
        self.sources_x = sources_x
        self.nz_padded = model_padded2_dt2.shape[0]
        self.nx_padded = model_padded2_dt2.shape[1]
        self.nzx_padded = self.nz_padded * self.nx_padded
        self.a1= a1
        self.a2= a2
        self.a22 =a22
        self.a21 =a21
        self.a3= a3
        self.a31 =a31
        self.a5 =a5
        self.a51= a51
        self.a4 =a4
        self.a41 =a41
        self.a42= a42
        self.a6 =a6
        self.a61= a61
        self.a62 =a62
        self.a7 =a7
        self.a71= a71
        self.a8 =a8
        self.a81= a81
        self.a9 =a9
        self.a91= a91
        self.a10 =a10
        self.a101= a101
        self.a11 =a11
        self.a111= a111
        self.a12 =a12
        self.a121= a121
        self.a13 =a13
        self.a131= a131
        self.a14 =a14
        self.a141= a141
        self.a15=a15
    @property
    def state_size(self):
        """The RNN state (passed between RNN units) contains two time steps
        of the wave field, and the PML auxiliary wavefields phiz and phix.
        """
        return [self.nzx_padded, self.nzx_padded,
                self.nzx_padded, self.nzx_padded,
                self.nzx_padded, self.nzx_padded, 
                self.nzx_padded, self.nzx_padded]

    @property
    def output_size(self):
        """The output of the RNN unit contains one time step of the wavefield.
        """
        return self.nzx_padded

    def __call__(self, inputs, state):
        """Propagate the wavefield forward one time step.

        Args:
            inputs: An array containing the source amplitudes for this time
                    step
            state: A list containing the two previous wave field time steps
                   and the auxiliary wavefields phiz and phix

        Returns:
            output: The current wave field
            state: A list containing the current and one previous wave field
                   time steps and the updated auxiliary wavefields phiz and
                   phix
        """
        inputs_shape = tf.shape(state[0])
        batch_size = inputs_shape[0]
        model_shape = [batch_size, self.nz_padded, self.nx_padded]
        wavefieldt = tf.reshape(state[0], model_shape)#t-1
        wavefieldtx = tf.reshape(state[1], model_shape)#t-2
        wavefieldtz = tf.reshape(state[2], model_shape)#t-1

        wavefield = tf.reshape(state[3], model_shape)#t-2
        wavefieldx = tf.reshape(state[4], model_shape)#t-1
        wavefieldz = tf.reshape(state[5], model_shape)#t-2
        phiz1c = tf.reshape(state[6], model_shape)#phiz
        phix1c = tf.reshape(state[7], model_shape)#phix

        '''
        phiz1c = tf.reshape(state[6], model_shape)#phiz
        phix1c = tf.reshape(state[7], model_shape)#phix
        phiz2c = tf.reshape(state[8], model_shape)#phiz
        phix2c = tf.reshape(state[9], model_shape)#phix
        phiz3c = tf.reshape(state[10], model_shape)#phiz
        phix3c = tf.reshape(state[11], model_shape)#phix
        '''
        #phizc = tf.reshape(state[2], model_shape)#phiz
        #phixc = tf.reshape(state[3], model_shape)#phix
        #wavefieldc_x=self.first_x_deriv(wavefieldc)
        #wavefieldc_z=self.first_z_deriv(wavefieldc)
        #wavefieldc_z=tf.expand_dims(wavefieldc_z ,0)   
        #wavefieldc_x=tf.expand_dims(wavefieldc_x ,0)  
        #lap = self.laplacian(wavefieldc)        
        #wavefieldc_x=tf.expand_dims(wavefieldc_x,-1)
        #wavefieldc_z=tf.expand_dims(wavefieldc_z,-1)
        lap=self.a2(wavefield)+self.a21(wavefieldx)+self.a22(wavefieldz)#ok      
        u3x=self.a3(wavefield)+self.a31(wavefieldx)#ok
        u3z=self.a5(wavefield)+self.a51(wavefieldz)#ok
        u2xz=self.a4(wavefield)+self.a41(wavefieldx)+self.a42(wavefieldz)#pk
        ux2z=self.a6(wavefield)+self.a61(wavefieldz)+self.a62(wavefieldx)#ok
        lap1=u3x+ux2z
        lap2=u3z+u2xz
        phiz1c_z =self.first_z_deriv3(phiz1c)
        phix1c_x = self.first_x_deriv3(phix1c)
   




        # The main evolution equation:
        '''
        s1=self.c2*self.d1*self.model_padded2_dt2
        s2=self.c1*self.c2*self.d1*self.model_padded2_dt2*self.dt
        s3=(self.d1+self.d2)*self.model_padded2_dt2/self.dt
        s4=self.d2*self.c2*self.d1*self.model_padded2_dt2**2/self.dt
        s5=(self.c1*self.d1+self.d2*self.c1+self.d2*self.c2)*self.model_padded2_dt2
        s6=self.d2*self.d1*self.c1*self.c2*self.model_padded2_dt2**2
        
        wavefieldtf = s3*(self.a2(wavefieldt)+self.a21(wavefieldtx)+self.a22(wavefieldtz))+s4*
        wavefieldtfx=wavefieldtx+1/2*self.model_padded2_dt2*(self.a3(wavefieldt)+self.a31(wavefieldtx)\
                             +self.a6(wavefieldt)+self.a61(wavefieldtz)+self.a62(wavefieldtx))+(self.model_padded2_dt2/self.dt*(u3x+ux2z)\
                                                                             +1/4/self.dt*self.model_padded2_dt2**2*(u5x+2*u3x2z+ux4z))
        wavefieldtfz= wavefieldtz+1/2*self.model_padded2_dt2*(self.a4(wavefieldt)+self.a41(wavefieldtx)+self.a42(wavefieldtz)\
                             +self.a5(wavefieldt)+self.a51(wavefieldtz))+(self.model_padded2_dt2/self.dt*(u2xz+u3z)\
                                                                             +1/4/self.dt*self.model_padded2_dt2**2*(u4xz+2*u2x3z+u5z))
        wavefieldf = wavefield+s1*lap+self.dt*wavefieldt+s2*(self.a2(wavefieldt)+self.a21(wavefieldtx)+self.a22(wavefieldtz))
        wavefieldfx= wavefieldx+s1*lap1+self.dt*wavefieldtx+s2*(self.a3(wavefieldt)+self.a31(wavefieldtx)+\
                                                                self.a6(wavefieldt)+self.a61(wavefieldtz)+self.a62(wavefieldtx))
        wavefieldfz =wavefieldz+s1*lap2+self.dt*wavefieldtz+s2*(self.a5(wavefieldt)+self.a51(wavefieldtz)+\
                                                                self.a4(wavefieldt)+self.a41(wavefieldtx)+self.a42(wavefieldtz))
        '''
        u11=wavefield+self.c1*self.dt*(wavefieldt-self.sigma_sum*wavefield)  
        u12=wavefieldx+self.c1*self.dt*(wavefieldtx-self.sigma_sum*wavefieldx)
        u13=wavefieldz+self.c1*self.dt*(wavefieldtz-self.sigma_sum*wavefieldz)
        
 
        
        v11=wavefieldt+self.d1*self.model_padded2_dt2/self.dt*(self.a2(u11)+self.a21(u12)+self.a22(u13)+phiz1c_z + phix1c_x)-self.d1*self.dt*self.sigma_prod_dt2*u11
        v12=wavefieldtx+self.d1*self.model_padded2_dt2/self.dt*(self.a3(u11)+self.a31(u12)+self.a6(u11)+self.a61(u13)+self.a62(u12))-self.d1*self.dt*self.sigma_prod_dt2*u12
        v13=wavefieldtz+self.d1*self.model_padded2_dt2/self.dt*(self.a5(u11)+self.a51(u13)+self.a4(u11)+self.a41(u12)+self.a42(u13))-self.d1*self.dt*self.sigma_prod_dt2*u13
        
        u21=u11+self.c2*self.dt*(v11-self.sigma_sum*u11)
        u22=u12+self.c2*self.dt*(v12-self.sigma_sum*u12)
        u23=u13+self.c2*self.dt*(v13-self.sigma_sum*u13)
        
  
        
        v21=v11+self.d2*self.model_padded2_dt2/self.dt*(self.a2(u21)+self.a21(u22)+self.a22(u23)+phiz1c_z + phix1c_x)-self.d2*self.dt*self.sigma_prod_dt2*u21
        v22=v12+self.d2*self.model_padded2_dt2/self.dt*(self.a3(u21)+self.a31(u22)+self.a6(u21)+self.a61(u23)+self.a62(u22))-self.d2*self.dt*self.sigma_prod_dt2*u22
        v23=v13+self.d2*self.model_padded2_dt2/self.dt*(self.a5(u21)+self.a51(u23)+self.a4(u21)+self.a41(u22)+self.a42(u23))-self.d2*self.dt*self.sigma_prod_dt2*u23
    
        u31=u21+self.c3*self.dt*(v21-self.sigma_sum*u21)
        u32=u22+self.c3*self.dt*(v22-self.sigma_sum*u22)
        u33=u23+self.c3*self.dt*(v23-self.sigma_sum*u23)
        
        
        v31=v21+self.d3*self.model_padded2_dt2/self.dt*(self.a2(u31)+self.a21(u32)+self.a22(u33)+phiz1c_z + phix1c_x)-self.d3*self.dt*self.sigma_prod_dt2*u31
        v32=v22+self.d3*self.model_padded2_dt2/self.dt*(self.a3(u31)+self.a31(u32)+self.a6(u31)+self.a61(u33)+self.a62(u32))-self.d3*self.dt*self.sigma_prod_dt2*u32
        v33=v23+self.d3*self.model_padded2_dt2/self.dt*(self.a5(u31)+self.a51(u33)+self.a4(u31)+self.a41(u32)+self.a42(u33))-self.d3*self.dt*self.sigma_prod_dt2*u33


        
        wavefieldf=u31+self.c4*self.dt*(v31-self.sigma_sum*u31)
        wavefieldfx=u32+self.c4*self.dt*(v32-self.sigma_sum*u32)
        wavefieldfz=u33+self.c4*self.dt*(v33-self.sigma_sum*u33)
        

        
        wavefieldtf=v31+self.d4*self.model_padded2_dt2/self.dt*(self.a2(wavefieldf)+self.a21(wavefieldfx)+self.a22(wavefieldfz)+phiz1c_z + phix1c_x)-self.d4*self.dt*self.sigma_prod_dt2*wavefieldf
        wavefieldtfx=v32+self.d4*self.model_padded2_dt2/self.dt*(self.a3(wavefieldf)+self.a31(wavefieldfx)+self.a6(wavefieldf)+self.a61(wavefieldfz)+self.a62(wavefieldfx))-self.d4*self.dt*self.sigma_prod_dt2*wavefieldfx
        wavefieldtfz=v33+self.d4*self.model_padded2_dt2/self.dt*(self.a4(wavefieldf)+self.a41(wavefieldfx)+self.a42(wavefieldfz)+self.a5(wavefieldf)+self.a51(wavefieldfz))-self.d4*self.dt*self.sigma_prod_dt2*wavefieldfz     
        
        # Update PML variables phix, phiz
        phiz1f = (phiz1c - self.dt * self.sigmaz * phiz1c
                 - self.dt * (self.sigmaz - self.sigmax) * wavefieldfz)
        phix1f = (phix1c - self.dt * self.sigmax * phix1c 
                 - self.dt * (self.sigmax - self.sigmaz) * wavefieldfx)
  
        # Add the sources
        # f(t+1, z_s, x_s) += c(z_s, x_s)^2 * dt^2 * s(t)
        # We need to expand "inputs" to be the same size as f(t+1), so we
        # use tf.scatter_nd. This will create an array
        # of the right size, almost entirely filled with zeros, with the
        # source amplitudes (multiplied by c^2 * dt^2) in the right places.
        wavefieldf += tf.scatter_nd(self.sources_x, inputs, model_shape)
        #wavefieldf_x += tf.scatter_nd(self.sources_x, inputs, model_shape)
        #wavefieldf_z += tf.scatter_nd(self.sources_x, inputs, model_shape)

        return (tf.reshape(wavefieldf, inputs_shape),
                [tf.reshape(wavefieldtf, inputs_shape),
                 tf.reshape(wavefieldtfx, inputs_shape),
                 tf.reshape(wavefieldtfz, inputs_shape),
                 tf.reshape(wavefieldf, inputs_shape),
                 tf.reshape(wavefieldfx, inputs_shape),
                 tf.reshape(wavefieldfz, inputs_shape),
                 tf.reshape(phiz1f, inputs_shape),
                 tf.reshape(phix1f, inputs_shape)])
    
  

def forward2d(model, sources, sources_x,
              dx, dt, pml_width=None, pad_width=None,
              profile=None):
    """Forward modeling using the 2D wave equation.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source
        dx: float specifying size of each cell (dx == dz)
        dt: float specifying time between time steps
        pml_width: number of cells in PML (optional)
        pad_width: number of padding cells outside PML (optional)
        profile: 1D array specifying PML profile (optional)

    Returns:
        4D Tensor [num_time_steps, batch_size, nz, nx] containing time steps of
        wavefields. Padding that was added is removed.
    """

    if pml_width is None:
        pml_width = 10
    if pad_width is None:
        pad_width = 8

    total_pad = pml_width + pad_width

    nz_padded, nx_padded = _set_x(model, total_pad)

    model_padded2_dt2 = _set_model(model, total_pad, dt)

    profile, pml_width = _set_profile(profile, pml_width, dx)

    sigmaz, sigmax = _set_sigma(nz_padded, nx_padded, total_pad, pad_width,
                                profile)

    sources, sources_x = _set_sources(sources, sources_x, total_pad,
                                      model_padded2_dt2)

    d1_kernel3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15= _set_kernels3(dx)

    first_z_deriv3, first_x_deriv3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15 = _set_deriv_funcs3(d1_kernel3,
                                                               a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15)

    cell = TimeStepCell(model_padded2_dt2, dt, sigmaz,sigmax,
                        first_z_deriv3, first_x_deriv3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15, sources_x)

    out, _ = tf.nn.dynamic_rnn(cell, sources,
                               dtype=tf.float32, time_major=True)

    out = tf.reshape(out, [int(out.shape[0]), # time
                           tf.shape(out)[1], # batch
                           nz_padded,
                           nx_padded])

    return out[:, :, total_pad : -total_pad, total_pad : -total_pad]


def _set_x(model, total_pad):
    """Calculate the size of the model after padding has been added.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge

    Returns:
        Integers specifying number of cells in padded model in z and x
    """
    nz = int(model.shape[0])
    nx = int(model.shape[1])
    nz_padded = nz + 2 * total_pad
    nx_padded = nx + 2 * total_pad
    return nz_padded, nx_padded


def _set_model(model, total_pad, dt):
    """Add padding to the model (extending edge values) and compute c^2 * dt^2.

    TensorFlow does not provide the option to extend the edge values into
    the padded region (unlike Numpy, which has an 'edge' option to do this),
    so we need to split the 2D array into 1D columns, pad the top with
    the first value from the column, and pad the bottom with the final value
    from the column, and then repeat it for rows.

    Args:
        model: 2D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge
        dt: Float specifying time step size

    Returns:
        A 2D Tensor containing the padded, squared model times the squared
        time step size
    """
    def pad_tensor(tensor, axis, pad_width):
        """Split the 2D Tensor into rows/columns along the specified axis, then
        iterate through those rows/columns padding the beginning and end with
        the first and last elements from the row/column. Then recombine back
        into a 2D Tensor again.
        """
        tmp1 = []
        for row in tf.unstack(tensor, axis=axis):
            tmp2 = tf.pad(row, [[pad_width, 0]], 'CONSTANT',
                          constant_values=row[0])
            tmp2 = tf.pad(tmp2, [[0, pad_width]], 'CONSTANT',
                          constant_values=row[-1])
            tmp1.append(tmp2)
        return tf.stack(tmp1, axis=axis)

    model_padded = pad_tensor(model, 0, total_pad)
    model_padded = pad_tensor(model_padded, 1, total_pad)
    return tf.square(model_padded) * dt**2


def _set_profile(profile, pml_width, dx):
    """Create a profile for the PML.

    Args:
        profile: User supplied profile, if None use default
        pml_width: Integer. If profile is None, create a PML of this width.
        dx: Float specifying spacing between grid cells

    Returns:
        profile: 1D array containing PML profile
        pml_width: Integer specifying the length of the profile
    """
    # This should be set to approximately the maximum wave speed at the edges
    # of the model
    max_vel = 8000
    if profile is None:
        profile =((np.arange(pml_width)/pml_width)**2
                   * 3 * max_vel * np.log(1000)
                   / (2 * dx * pml_width))
    else:
        pml_width = len(profile)
    return profile, pml_width


def _set_sigma(nz_padded, nx_padded, total_pad, pad_width, profile):
    """Create 1D sigma arrays that contain the PML profile in the PML regions.

    Args:
        nz_padded: Integer specifying the number of depth cells in the padded
                   model
        nx_padded: Integer specifying the number of x cells in the padded model
        total_pad: Integer specifying the number of cells of padding added to
                   each edge of the model
        pad_width: Integer specifying the number of cells of padding that are
                   not part of the PML
        profile: 1D array containing the PML profile for the bottom/right side
                 of the model (for the top/left side, it will be reversed)

    Returns:
        1D sigma arrays for the depth and x directions
    """
    def sigma_1d(n_padded, total_pad, pad_width, profile):
        """Create one 1D sigma array."""
        sigma = np.zeros(n_padded, np.float32)
        sigma[total_pad-1:pad_width-1:-1] = profile
        sigma[-total_pad:-pad_width] = profile
        sigma[:pad_width] = sigma[pad_width]
        sigma[-pad_width:] = sigma[-pad_width-1]
        return sigma

    sigmaz = sigma_1d(nz_padded, total_pad, pad_width, profile)
    sigmaz = sigmaz.reshape([-1, 1])
    sigmaz = np.tile(sigmaz, [1, nx_padded])

    sigmax = sigma_1d(nx_padded, total_pad, pad_width, profile)
    sigmax = sigmax.reshape([1, -1])
    sigmax = np.tile(sigmax, [nz_padded, 1])

    return tf.constant(sigmaz), tf.constant(sigmax)

def _set_sources(sources, sources_x, total_pad, model_padded2_dt2):
    """Set the source amplitudes, and the source positions.

    Args:
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 3]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer z cell coordinate of
                   the source, and [:, :, 2] contains the integer x cell
                   coordinate of the source
        total_pad: Integer specifying padding added to each edge of the model
        model_padded2_dt2: Tensor containing squared wave speed times squared
                           time step size

    Returns:
        sources: 3D Tensor containing source amplitude * c^2 * dt^2
        sources_x: 3D Tensor like the input, but with total_pad added to
                   [:, :, 1] and [:, :, 2]
    """
    # I add "total_pad" to the source coordinates as the coordinates currently
    # refer to the coordinates in the unpadded model, but we need them to
    # refer to the coordinates when padding has been added. We only want to add
    # this to [:, :, 1] and [:, :, 2], which contains the depth and x
    # coordinates, so I multiply by an array that is 0 for [:, :, 0], and 1
    # for [:, :, 1] and [:, :, 2].
    sources_x += (tf.ones_like(sources_x) * total_pad
                  * np.array([0, 1, 1]).reshape([1, 1, 3]))

    # The propagator injected source amplitude multiplied by c(x)^2 * dt^2
    # at the locations of the sources, so we need to extract the wave speed
    # at these locations. I do this using tf.gather
    sources_v = tf.gather_nd(model_padded2_dt2, sources_x[:, :, 1:])

    # The propagator does not need the unmultiplied source amplitudes,
    # so I will save space by only storing the source amplitudes multiplied
    # by c(x)^2 * dt^2
    sources = sources * sources_v

    return sources, sources_x


def _set_kernels3(dx):
    """Create spatial finite difference kernels.

    The kernels are reshaped into the appropriate shape for a 2D
    convolution, and saved as constant tensors.

    Args:
        dx: Float specifying the grid cell spacing

    Returns:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 2D second derivative (Laplacian)
    """
    # First derivative
    d1_kernel = (np.array([1/12, -2/3, 0, 2/3, -1/12], np.float32)
                 / dx)
    
    a2=(np.array([[0.0, 0,7/54,0, 0.0],
                  [0.0, 0,64/27,0, 0.0],
                  [7/54, 64/27,-10,64/27, 7/54],#2x+2z
                  [0.0, 0,64/27,0, 0.0],
                  [0.0, 0,7/54,0, 0.0]],np.float32)/dx/dx)
    
    a21=(np.array( [[0.0, 0.0,0, 0.0,0],
                    [0.0, 0.0, 0,0.0,0],
                    [1/36, 8/9,0, -8/9,-1/36],
                    [0.0, 0.0, 0,0.0,0],
                    [0.0, 0.0,0, 0.0,0]], np.float32)/ dx)
    a22=a21.T
    a3=(np.array( [[0.0, 0.0,0, 0.0,0],
                    [0.0, 0.0, 0,0.0,0],
                    [-31/144, -88/9,0, 88/9,31/144],
                    [0.0, 0.0, 0,0.0,0],
                    [0.0, 0.0,0, 0.0,0]], np.float32)/ dx/dx/dx) 
    a31=(np.array( [[0.0, 0.0,0, 0.0,0],
                    [0.0, 0.0, 0,0.0,0],
                    [-1/24, -8/3,-15, -8/3,-1/24],
                    [0.0, 0.0, 0,0.0,0],
                    [0.0, 0.0,0, 0.0,0]], np.float32)/ dx/dx) 
    a5=a3.T
    a51=a31.T
    a4=(np.array( [[-31/864/dx/dx/dx, 0.0,2*31/864/dx/dx/dx, 0,-31/864/dx/dx/dx],
                    [0.0, -44/27/dx/dx/dx, 2*44/27/dx/dx/dx,-44/27/dx/dx/dx,0],
                    [0, 0,0, 0,0],
                    [0.0, 44/27/dx/dx/dx, -2*44/27/dx/dx/dx,44/27/dx/dx/dx,0],
                    [31/864/dx/dx/dx, 0.0,-2*31/864/dx/dx/dx,0,31/864/dx/dx/dx]], np.float32)) 
    a41=(np.array( [[-1/144/dx/dx, 0.0,0, 0.0,1/144/dx/dx],
                    [0.0, -4/9/dx/dx, 0,4/9/dx/dx,0],
                    [0, 0,0, 0,0],
                    [0.0, 4/9/dx/dx, 0,-4/9/dx/dx,0],
                    [1/144/dx/dx, 0.0,0, 0.0,-1/144/dx/dx]], np.float32)/ 1) 
    a42=(np.array( [[-1/144/dx/dx, 0.0,2/144/dx/dx, 0.0,-1/144/dx/dx],
                    [0.0, -4/9/dx/dx, 8/9/dx/dx,-4/9/dx/dx,0],
                    [0, 0,0, 0,0],
                    [0.0, -4/9/dx/dx, 8/9/dx/dx,-4/9/dx/dx,0],
                    [-1/144/dx/dx, 0,2/144/dx/dx, 0,-1/144/dx/dx]], np.float32)/1)    
    a6=a4.T#2zx

    a61=a41.T
    a62=a42.T
    a7=(np.array([[0, 0, 0],
                   [1, -2, 1],
                   [0, 0, 0]], np.float32)/ dx**4*-12)  #4x
    a71=(np.array( [[0.0, 0.0, 0.0],
                    [-1, 0, 1],
                    [0.0, 0.0, 0.0]], np.float32)/ dx**3*-6) #4x   
    a8=a7.T#4z
    a81=a71.T
    a9=(np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]], np.float32)/ dx**4/dx*-90) 
    a91=(np.array([[0, 0, 0],
                   [1, 4, 1],
                   [0, 0, 0]], np.float32)/ dx**4*30) 
    a10=a9.T
    a101=a91.T

    a11=(np.array([[-1, 0, 1],
                  [2, 0, -2],
                  [-1, 0, 1]], np.float32)/ dx**4/dx*3)
    a111=(np.array([[0, 1, 0],
                  [0, -2, 0],
                  [0, 1, 0]], np.float32)/ dx**4*-6)
    a12=a11.T#2x3z
    a121=a111.T
    a13=(np.array([[-5, 6, -1],
                  [4, 0, -4],
                  [1, -6, 5]], np.float32)/ dx**4/dx*-3)#x4z
    a131=(np.array([[1, -1, 0],
                  [2, -4, 2],
                  [0, -1, 1]], np.float32)/ dx**4*6)
    a14=a13.T#4xz
    a141=a131.T
    a1=(np.array([[1.0, -2.0, 1.0],
                  [-2.0, 4.0, -2.0],
                  [1.0, -2.0, 1.0]], np.float32)/ dx**4)#2x2z
    a15=(np.array([[1, 0, -1],
                  [0, 0, 0],
                  [-1, 0, 1]], np.float32)/ dx/dx/4)

    #d2_kernel=a1*2-a2*d1_kernel
    '''
    a1=(np.array([[0.0,   0.0, 7/54, 0.0, 0.0],
                          [0.0,   0.0, 64/27,   0.0, 0.0],
                          [7/54, 64/27, -10,  64/27, 7/54],
                          [0.0,   0.0, 64/27,   0.0, 0.0],
                          [0.0,   0.0, 7/54, 0.0, 0.0]],
                         np.float32)/dx**2)
    a2=(np.array([1/36,8/9,0, 8/9, 1/36], np.float32)
                 / dx)
    a3=(np.array([1/36,8/9,0, 8/9, 1/36], np.float32)
                 / dx)
    '''
    
    d1_kernel3 = tf.constant(d1_kernel)
    #d3_kernel = tf.constant(d3_kernel)
    #d4_kernel = tf.constant(d4_kernel)


    # Second derivative
    '''
    d2_kernel = np.array([[0.0,   0.0, -1/12, 0.0, 0.0],
                          [0.0,   0.0, 4/3,   0.0, 0.0],
                          [-1/12, 4/3, -10/2,  4/3, -1/12],
                          [0.0,   0.0, 4/3,   0.0, 0.0],
                          [0.0,   0.0, -1/12, 0.0, 0.0]],
                         np.float32)
    '''
    #d2_kernel = np.array([[0.0, 0.0,0.0, 1/90, 0.0, 0.0, 0.0],
                        #  [0.0, 0.0, 0.0,-3/20,0.0, 0.0, 0.0],
                        #  [0.0, 0.0,0.0,  3/2, 0.0, 0.0, 0.0],
                         # [1/90,-3/20, 3/2,-49/9, 3/2, -3/20, 1/90],
                        #  [0.0, 0.0 , 0.0, 3/2,   0.0, 0.0, 0.0],
                        #  [0.0,  0.0, 0.0,-3/20, 0.0, 0.0, 0.0],
                         # [0.0, 0.0,  0.0, 1/90, 0.0, 0.0, 0.0]],
                        # np.float32)
    #d2_kernel /= dx**2
    a1 = tf.constant(a1.reshape([3, 3, 1, 1]))
    a2 = tf.constant(a2.reshape([5, 5, 1, 1]))
    a21 = tf.constant(a21.reshape([5, 5, 1, 1]))
    a22 = tf.constant(a22.reshape([5, 5, 1, 1]))    
    a3 = tf.constant(a3.reshape([5, 5, 1, 1]))
    a31 = tf.constant(a31.reshape([5, 5, 1, 1]))
    a5 = tf.constant(a5.reshape([5, 5, 1, 1]))
    a51 = tf.constant(a51.reshape([5, 5, 1, 1]))
    a4 = tf.constant(a4.reshape([5, 5, 1, 1]))
    a41 = tf.constant(a41.reshape([5, 5, 1, 1]))
    a42 = tf.constant(a42.reshape([5, 5, 1, 1]))
    a6 = tf.constant(a6.reshape([5, 5, 1, 1]))
    a61 = tf.constant(a61.reshape([5, 5, 1, 1]))
    a62 = tf.constant(a62.reshape([5, 5, 1, 1]))  
    a7 = tf.constant(a7.reshape([3, 3, 1, 1]))
    a71 = tf.constant(a71.reshape([3, 3, 1, 1])) 
    a8 = tf.constant(a8.reshape([3, 3, 1, 1]))
    a81 = tf.constant(a81.reshape([3, 3, 1, 1]))
    a9 = tf.constant(a9.reshape([3, 3, 1, 1]))
    a91 = tf.constant(a91.reshape([3, 3, 1, 1]))
    a10 = tf.constant(a10.reshape([3, 3, 1, 1]))
    a101 = tf.constant(a101.reshape([3, 3, 1, 1]))
    a11 = tf.constant(a11.reshape([3, 3, 1, 1]))
    a111 = tf.constant(a111.reshape([3, 3, 1, 1]))
    a12 = tf.constant(a12.reshape([3, 3, 1, 1]))
    a121 = tf.constant(a121.reshape([3, 3, 1, 1]))
    a13 = tf.constant(a13.reshape([3, 3, 1, 1]))
    a131 = tf.constant(a131.reshape([3, 3, 1, 1]))  
    a14 = tf.constant(a14.reshape([3, 3, 1, 1]))
    a141 = tf.constant(a141.reshape([3, 3, 1, 1]))
    a15 = tf.constant(a15.reshape([3, 3, 1, 1]))
    return d1_kernel3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15


def _set_deriv_funcs3(d1_kernel3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15):
    """Create functions to apply first and second derivatives.

    Args:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 2D second derivative (Laplacian)

    Returns:
        Functions for applying first (in depth and x) and second derivatives
    """
    def make_deriv_func(kernel, shape):
        """Returns a function that takes a derivative of its input."""
        def deriv(x):
            """Take a derivative of the input."""
            return tf.squeeze(tf.nn.conv2d(tf.expand_dims(x, -1),
                                           tf.reshape(kernel, shape),
                                           [1, 1, 1, 1], 'SAME'))
        return deriv

    first_z_deriv3 = make_deriv_func(d1_kernel3, [-1, 1, 1, 1])
    first_x_deriv3 = make_deriv_func(d1_kernel3, [1, -1, 1, 1])
    #laplacian = make_deriv_func(d2_kernel, d2_kernel.shape)
    a1= make_deriv_func(a1, a1.shape)
    a2= make_deriv_func(a2, a2.shape)
    a21= make_deriv_func(a21, a21.shape)
    a22= make_deriv_func(a22, a22.shape)
    a3= make_deriv_func(a3, a3.shape)
    a31= make_deriv_func(a31, a31.shape)
    a5= make_deriv_func(a5, a5.shape)
    a51= make_deriv_func(a51, a51.shape)
    a4= make_deriv_func(a4, a4.shape)
    a41= make_deriv_func(a41, a41.shape)
    a42= make_deriv_func(a42, a42.shape)
    a6= make_deriv_func(a6, a6.shape)
    a61= make_deriv_func(a61, a61.shape)
    a62= make_deriv_func(a62, a62.shape)
    a7= make_deriv_func(a7, a7.shape)
    a71= make_deriv_func(a71, a71.shape)   
    a8= make_deriv_func(a8, a8.shape)
    a81= make_deriv_func(a81, a81.shape)
    a9= make_deriv_func(a9, a9.shape)
    a91= make_deriv_func(a91, a91.shape)
    a10= make_deriv_func(a10, a10.shape)
    a101= make_deriv_func(a101, a101.shape)
    a11= make_deriv_func(a11, a11.shape)
    a111= make_deriv_func(a111, a111.shape)
    a12= make_deriv_func(a12, a12.shape)
    a121= make_deriv_func(a121, a121.shape)
    a13= make_deriv_func(a13, a13.shape)
    a131= make_deriv_func(a131, a131.shape)
    a14= make_deriv_func(a14, a14.shape)
    a141= make_deriv_func(a141, a141.shape)
    a15= make_deriv_func(a15, a15.shape)
    return first_z_deriv3, first_x_deriv3, a1,a2,a21,a22,a3,a31,a5,a51,a4,a41,a42,a6,a61,a62,a7,a71,a8,a81,a9,a91,a10,a101,a11,a111,a12,a121,a13,a131,a14,a141,a15
#laplacian
