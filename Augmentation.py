import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
SETTING_W = 224
SETTING_H = 224

def inv_polar(data):
    """
    Invert the polarity of events in the given data.

    Parameters
    ----------
    data : structured numpy array
        The events to be inverted.
        data neet obeys the following structure:
        (1) structured numpy array with fields: 't', 'x', 'y', 'p'
        (2) 't' is the timestamp
        (3) 'x' and 'y' are the coordinates
        (4) 'p' is the polarity
    """
    data['p'] = 1-data['p']
    return data
def filp(data,type):
    """
    Flip the coordinates of events in the given data.

    Parameters
    ----------
    data : structured numpy array
        The events to be flipped. It should have fields 'x' and 'y' for coordinates.
    type : str
        The type of flip to perform. 'h' for horizontal flip and 'v' for vertical flip.

    """
    global SETTING_H,SETTING_W
    if type == 'h':
        data['x'] = SETTING_W-data['x']
    elif type == 'v':
        data['y'] = SETTING_H-data['y']
    else:
        print('type error')
def rotation(data,theta):
    """
    Rotate the coordinates of events in the given data.

    Parameters
    ----------
    data : structured numpy array
        The events to be rotated.
        data neet obeys the following structure:
        (1) structured numpy array with fields: 't', 'x', 'y', 'p'
        (2) 't' is the timestamp
        (3) 'x' and 'y' are the coordinates
        (4) 'p' is the polarity
    theta : float, the angle of rotation in radians
        The angle of rotation in radians, positive for counter-clockwise rotation.
    """
    center_x = SETTING_W/2 # calcualte the centrol of the image
    center_y = SETTING_H/2
    new_x = data['x'] - center_x
    new_y = data['y'] - center_y
    # new coordinates after rotation (base on the center of the image)
    new_x_r = np.floor(new_x * np.cos(theta) - new_y * np.sin(theta) + center_x) 
    new_y_r = np.floor(new_x * np.sin(theta) + new_y * np.cos(theta) + center_y) 
    # clip the new coordinates / remapping the new coordinates back to the original image coordinate system
    new_x = np.clip(new_x_r, 0, SETTING_W - 1)
    new_y = np.clip(new_y_r, 0, SETTING_H - 1)
    data['x'] = new_x
    data['y'] = new_y
    
    return data
def viz_events(events,inv=False):
    """
    Visualize events as an image.

    Parameters
    ----------
    events : structured numpy array
        The events to be visualized.
        events neet obeys the following structure:
        (1) structured numpy array with fields: 't', 'x', 'y', 'p'
        (2) 't' is the timestamp
        (3) 'x' and 'y' are the coordinates
        (4) 'p' is the polarity
    inv : bool, optional
        If True, the polarity of events is inverted.
    """
    img = np.full((SETTING_H, SETTING_W, 3), 128, dtype=np.uint8)
    if inv:
        img[events['y'], events['x']] = 255 - 255 * events['p'][:, None]
    else:
        img[events['y'], events['x']] = 255 * events['p'][:, None]
    return img

if __name__ == '__main__':
    # 1. load data path
    pathlist = glob('./*.npy')
    dir = os.path.dirname(pathlist[0])
    for idx,path in enumerate(pathlist):
        name=os.path.basename(path)
        data = np.load(path)
        if idx == 0:
            image = viz_events(data)
            newname = name.split('.')[0] + '.png'
            plt.savefig(os.path.join(dir,newname))
        
        #inv_polar
        data = inv_polar(data)
        newname = name.split('.')[0] + '_inv_polar.npy'
        newpath = os.path.join(dir,newname)
        np.save(newpath,data)
        if idx == 0:
            image = viz_events(data)
            newname = name.split('.')[0] + '_inv_polar.png'
            plt.savefig(os.path.join(dir,newname))
        #90
        data =  rotation(data,90)
        newname = name.split('.')[0] + '_90rotated.npy'
        newpath = os.path.join(dir,newname)
        np.save(newpath,data)
        if idx == 0:
            image = viz_events(data)
            newname = name.split('.')[0] + '_90rotated.png'
            plt.savefig(os.path.join(dir,newname))
        #180 
        data =  rotation(data,180)
        newname = name.split('.')[0] + '_180rotated.npy'
        newpath = os.path.join(dir,newname)
        np.save(newpath,data)
        if idx == 0:
            image = viz_events(data)
            newname = name.split('.')[0] + '_180rotated.png'
            plt.savefig(os.path.join(dir,newname))
        #270
        data =  rotation(data,270)
        newname = name.split('.')[0] + '_270rotated.npy'
        newpath = os.path.join(dir,newname)
        np.save(newpath,data)
        if idx == 0:
            image = viz_events(data)
            newname = name.split('.')[0] + '_270rotated.png'
            plt.savefig(os.path.join(dir,newname))
    print(f'finish,directory: {dir}')
        
        
       
        
        