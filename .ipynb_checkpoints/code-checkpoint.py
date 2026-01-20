import numpy as np
import h5py

def sdss_rgb(imgs, bands=["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)),m = 0.03):
    '''
    Function to convert the 3x128x128 images into RGB images using a color transformation
    '''
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def read_h5_random_subset(h5_path,n_sample,seed=0):
    """
    Read a random subset from an HDF5 galaxy image dataset.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file.
    n_sample : int
        Number of objects to sample.
    seed : int
        Random seed.

    Returns
    -------
    images : np.ndarray
        Shape (n_sample, 3, 128, 128) depending on file.
    targetid : np.ndarray
    ra : np.ndarray
    dec : np.ndarray
    zred : np.ndarray
    """

    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]

        if n_sample > N:
            raise ValueError("n_sample exceeds dataset size")

        inds = rng.choice(N, size=n_sample, replace=False)
        inds.sort()  # improves HDF5 read performance

        images = f["images"][inds]
        targetid = f["TARGETID"][inds]
        ra = f["RA"][inds]
        dec = f["DEC"][inds]
        zred = f["Z"][inds]

    return images, targetid, ra, dec, zred

