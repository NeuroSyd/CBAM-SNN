from dataclasses import dataclass
import multiprocessing.pool as mpool
import os
from expelliarmus import Wizard
import numpy as np
import time
from multiprocessing import Pool
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# to disable multiprocessing, set N_PROCESSORS to 1
N_PROCESSORS = 1
SENSOR_SIZE = (1280, 720, 2,)
Height = 720
Width = 1280
df_f = 1
Pol_inv = False
roix =[int(0.225*Width)//df_f,int(0.4*Width)//df_f]
roiy = [int(0*Height)//df_f,int(1*Height)//df_f]

width = roix[1]-roix[0]
height = roiy[1] - roiy[0]

# file names
'''
Filelist = ['8um_2000D_10uL_50to10min.raw','3um_15uL_-25bias_-25off_25fo_lux+2.raw','15um_10uL_50to10min.raw',
            '3um_1uL_0bias.raw','3um_1uL_-25bias.raw','3um_5uL_-25bias.raw',
            '3um_10uL_0bias.raw','3um_10uL_-25bias.raw','3um_15uL_0bias.ra
            '3um_15uL_-25bias_-25off_25fo_lux+2.raw','THP1_0d_1uL_10min_01.raw'] 
'''
Filelist = ['platelet_40x_ph2_1_cut2_inv.raw','platelet_40x_ph2_1_cut2.raw'] #,'bd_40xph2_1ul_cut1_2.raw','bd_40xph2_1ul_cut1.raw','PMBC_40x_1ul_best_cut2_30s.raw','PMBC_40x_1ul_best_cut2_30s_inv.raw','NUE_40X1_PH2_cut2_30s_inv.raw','NUE_40X1_PH2_cut2_30s.raw'
#'3um_1uL_-25bias.raw','3um_5uL_-25bias.raw','3um_10uL_0bias.raw','3um_10uL_-25bias.raw','3um_15uL_0bias.raw',,'8um_2000D_10uL_50to10min.raw','15um_10uL_50to10min.raw','3um_1uL_0bias.raw','15um_10uL_50to10min.raw','3um_1uL_0bias.raw','3um_1uL_-25bias.raw','3um_5uL_-25bias.raw','3um_10uL_0bias.raw','3um_10uL_-25bias.raw','3um_15uL_0bias.raw','3um_15uL_-25bias_-25off_25fo_lux+2.raw'
#raw file folder'thp1_10d_0.5ul_20x.raw',
database = "../data/"
raw_base = "raw_pro/"

############################################# from paper code start****************
@dataclass
class Downsample:
    """Copied from tonic.transforms.Downsample. Removed events.copy() for
    (possibly) better memory efficiency."""

    spatial_factor: float = 1

    def __call__(self, events):
        if "x" in events.dtype.names:
            events["x"] = events["x"] * self.spatial_factor
        if "y" in events.dtype.names:
            events["y"] = events["y"] * self.spatial_factor
        return events


@dataclass
class LowPassLIF:
    """Low pass filter through simple LIF neuron model."""

    weight: float = 1.5
    vrest: float = 0.0
    vthr: float = 2.0
    leak: float = 0.5
    trefr: int = 1

    sensor_size: tuple = (int(width/df_f), int(height/df_f), 2)

    def __call__(self, events):
        # start event sorting
        ts = time.time()
        map_time_to_evidxs = {}
        for idx, evt in enumerate(events):
            if N_PROCESSORS == 1:
                if idx % int(1e5) == 0:
                    print(f"\revent sorting {idx/len(events):.2%}", end=" "*10)
            if evt['t'] in map_time_to_evidxs:
                map_time_to_evidxs[evt['t']].append(idx)
            else:
                map_time_to_evidxs[evt['t']] = [idx]
        print(f"\rfinished event sorting [{(time.time()-ts)/60:.1f}min]")

        # start LIF processing
        membrane = np.zeros(self.sensor_size, dtype=np.float32)
        event_times = np.unique(events['t'])
        last_updated_t = 0
        events_lpf = []
        refr = {}
        ts = time.time()
        for idx, evtt in enumerate(event_times):
            # log progress
            if N_PROCESSORS == 1:
                if idx % 100 == 0:
                    print(f"\rLIF {idx/len(event_times):.2%}", end=" "*10)
            # clean up old refractory periods
            for del_key in [tk for tk in refr if tk < (evtt-self.trefr-1)]:
                del refr[del_key]
            # update membrane potentials
            membrane = (self.leak ** (evtt - last_updated_t)) * membrane #
            # iterate over events at current timestep to check for resulting spikes
            for ev_idx in map_time_to_evidxs[evtt]:
                evt = events[ev_idx]
                (_,evtx,evty,evtp) = evt
                if evt in refr.get(evtt, []):
                    # ignore events if in refractory period
                    continue
                membrane[evtx,evty,evtp] += self.weight
                if membrane[evtx,evty,evtp] >= self.vthr:
                    # if spike, then add to events_lpf
                    membrane[evtx,evty,evtp] = self.vrest
                    events_lpf.append(evt)
                    # add refractory events
                    for i in range(self.trefr+1):
                        refr[i] = refr.get(i, []) + [(evtx,evty,evtp,evtt+i)]
        print(f'\rfinished LIF processing [{(time.time()-ts)/60:.1f}min]')
        return np.array(events_lpf, dtype=events.dtype)
    
############################################################################

FilebasePath = f'{database}{raw_base}'
wizard = Wizard(encoding="evt3")
name_list = Filelist

imageW = 224
imageH = 224

heightoffset = roiy[0]
widthoffset = roix[0]

def viz_events(events,inv=False):
    global imageW,imageH,heightoffset,widthoffset
    img = np.full((imageH, imageH, 3), 128, dtype=np.uint8)
    if inv:
        img[events['y'], events['x']] = 255 - 255 * events['p'][:, None]
    else:
        img[events['y'], events['x']] = 255 * events['p'][:, None]
    return img

        
############split by 100us ##########################################

split_list = glob.glob(f"{database}numpy/*.npy")

Sp_path_t = f"{database}listsp"
finish_num = mp.Value('i', 0)
skip_num = mp.Value('i', 0)
total_P_num = mp.Value('i', 0)
lock = mp.Lock()

def creat_split_point_list(durT,arr):
    durT = durT*2e2 #100us
    split_point_list = []
    bin_num = int(arr[-1]['t']/durT)
    for i in range(bin_num):
        start = arr[0]['t']
        startp = start + durT * i
        split_point_list.append(startp)
    return split_point_list

def basic_SP_Task(events,start,durT,save_path):
    unit = 1e2 #50us
    arr1 = events[(events['t']>=start) & (events['t']<(start+durT*unit))]
    print(f"start: {start}, durT: {durT}, arr1: {arr1.shape}")
    
    finish_num = 0
    total_P_num = 0
    skip_num = 0
    if len(arr1) >= 300:
        arr1 = crop_cal(arr1)
        file_path = os.path.join(save_path, f"{int((start-events[0]['t'])/unit)}00us.npy")
        pic_path = os.path.join(save_path, f"{int((start-events[0]['t'])/unit)}00us.png")
        img = viz_events(arr1,Pol_inv)
        plt.imsave(pic_path,img)
        np.save(file_path,arr1)
        finish_num =  1
        total_P_num = 1
    else:
        skip_num =  1
        total_P_num = 1
    return skip_num,finish_num,total_P_num
def update_log_file(filename, log_path="processed_files.log"):
    # Check if the log file exists; if not, create it
    if not os.path.exists(log_path):
        with open(log_path, 'w') as log_file:
            log_file.write("")  # Create the file by writing an empty string

    # Now that we're sure the file exists, append the filename
    with open(log_path, 'a') as log_file:
        log_file.write(filename + '\n')

# Define a function to check if a file has been processed
def is_file_processed(filename, log_path="processed_files.log"):
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            processed_files = log_file.readlines()
        return filename + '\n' in processed_files
    return False

def task_T(filename):
    if is_file_processed(filename, log_path=f"{database}numpysp2/processed_sp_files.log"):
            print(f"Skipping {filename} because it has been processed.")
    else:
        try:
            print(f'start procesing {filename}')
            arr = np.load(filename)
            print(f'{filename} shape: {arr.shape}')
            startlist = creat_split_point_list(durT=3,arr=arr)
            durT = 2
            skip_num = 0
            finish_num = 0
            total_P_num = 0
            save_path = f'{database}numpysp2/{filename.split("/")[-1].split(".")[0]}'
            print(save_path)
            def error_handler(error):
                print("Error in worker:", error)
            def update_counts(result):
                nonlocal skip_num, finish_num, total_P_num
                local_skip_num, local_finish_num, local_total_P_num = result
                skip_num += local_skip_num
                finish_num += local_finish_num
                total_P_num += local_total_P_num
                pbar.update(1) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            poolworker = mpool.ThreadPool(1)
            with tqdm(total=len(startlist), desc=f"Processing {filename}", position=0, leave=True) as pbar:
                for start in startlist:
                    poolworker.apply_async(basic_SP_Task, (arr, start, durT, save_path), callback=update_counts, error_callback=error_handler)
                poolworker.close()
                poolworker.join()
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        update_log_file(filename, log_path=f"{database}numpysp2/processed_sp_files.log")

def crop_cal(eventin):
    # Perform clustering
    kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as necessary
    x_coords = eventin['x']
    y_coords = eventin['y']
    coords =np.column_stack((x_coords, y_coords))
    kmeans.fit(coords)
    labels = kmeans.labels_
    
    # Find the largest cluster
    largest_cluster_idx = np.argmax(np.bincount(labels))
    largest_cluster = coords[labels == largest_cluster_idx]

    # Calculate the centroid of the largest cluster
    _, center_y = np.mean(largest_cluster, axis=0)

    # Crop the image around the centroid
    crop_half_size = 112  # Half of 244 for cropping
    topY = int(max(0, center_y - crop_half_size))
    downY = int(min(720, center_y + crop_half_size))
    print(f"topY: {topY}, downY: {downY}")
    if topY-downY < 224:
        downY = topY + 223
    cropped_img = eventin[(eventin['y'] >= topY) & (eventin['y'] <= downY)]
    cropped_img['y'] = cropped_img['y'] - topY
    return cropped_img

    
if __name__ == '__main__':
    for name in name_list:
        if is_file_processed(name, log_path=f"{database}numpy/processed_files.log"):
            print(f"Skipping {name} because it has been processed.")
            continue
        else:
            wizard.set_file(FilebasePath + name)
            arr = wizard.read()
            #ds_trf = Downsample(spatial_factor=1/df_f)
            
            #all_evs = lp_trf(ds_trf(arr))
            #arr = ds_trf(arr)
            arr = arr[(arr['x']>roix[0])&(arr['x']<roix[1])&(arr['y']>roiy[0])&(arr['y']<roiy[1])]
            arr['x'] = (arr['x']-widthoffset)//df_f
            arr['y'] = (arr['y']-heightoffset)//df_f
            #lp_trf = LowPassLIF(sensor_size=(width//df_f, height//df_f, 2))
            #ds_trf = Downsample(spatial_factor=1/df_f)
            #arr = lp_trf(ds_trf(arr))
            #arr = lp_trf(arr)
            #arr = crop_cal(arr)
            
            print(f"Recording duration: {(arr[-1]['t']-arr[0]['t'])/2e6:.2f} s.")
            print(f"{name}:", arr.shape)
            split = name.split('.')
            if not os.path.exists(f"{database}numpy/"):
                os.makedirs(f"{database}numpy/")
            np.save(f"{database}numpy/{split[0]}.npy", arr)
            update_log_file(name, log_path=f"{database}numpy/processed_files.log")
    split_list = glob.glob(f"{database}numpy/*.npy")
    if not split_list:
        print("Warning: split_list is empty.")
    else:
        worksPool = mpool.Pool(6)
        worksPool.map(task_T,split_list)
        worksPool.close()
        worksPool.join()
        print("All split processes done.")
    

        

    

        
