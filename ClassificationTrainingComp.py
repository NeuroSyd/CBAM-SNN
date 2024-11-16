import numpy as np
import os
import random
import torch
from PIL import Image
import torch.optim.sgd
import torchvision.transforms as transforms
from scipy.ndimage import zoom
import cv2
import glob
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import ClassificationModel as MD
import time
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import umap
from ClassificationModel import Cnnbase,ResidualBlock,TCnnbase,SNNBlockV3M2,SnnMLPLayer,SNNBlockV3M1,SNNBlockVS2,SnnResidualBlock,TCFATestBlock,MSM2,TCFATestBlockVCST,TCFATestBlockVTCS,TCFATestBlockVCSTP,TCFATestBlockVCSTPN
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import pickle  
from collections import defaultdict
from torch.amp import autocast, GradScaler
import snntorch as snn
import snntorch.functional as SF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import classification_report,precision_recall_curve,average_precision_score
from torch.optim import lr_scheduler
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages 
from pathlib import Path
# fixed seed
gseed = 2024
#dataset
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def direct_event_to_tensor(eventfile,type='npy',timesteps=4):
    #create one tensor from the event file,size is [t,c,w,h]
    data = np.zeros((timesteps,2,224,224))
    try:
        if os.path.exists(eventfile):
            if type == 'npy':
                event = np.load(eventfile)
            elif type == 'txt':
                event = np.loadtxt(eventfile)
            else:
                raise ValueError(f"Unknown file type: {type}")
        else:
            raise FileNotFoundError(f"File not found: {eventfile}")
        starttime = event['t'][0]
        unit_time  = np.floor(400/timesteps)
        t = event['t']
        t = ((t - starttime) // unit_time).astype(int)
        p = event['p']
        w = event['x']
        h = event['y']

        data[t, p, h, w] = 1
        return data
    except Exception as e:
        print(e)
        return None
def direct_event_to_tensor_optimized(eventfile, type='npy', timesteps=4):
    data = np.zeros((timesteps, 2, 224, 224), dtype=np.float32)
    try:
        if os.path.exists(eventfile):
            if type == 'npy':
                event = np.load(eventfile)
            elif type == 'txt':
                event = np.loadtxt(eventfile)
            else:
                raise ValueError(f"Unknown file type: {type}")
        else:
            raise FileNotFoundError(f"File not found: {eventfile}")
        starttime = event['t'][0]
        unit_time = np.floor(400 / timesteps).astype(np.float32)  # Ensure float32

        # Ensure event['t'] and starttime are float32
        event_t = event['t'].astype(np.float32)
        starttime = np.float32(starttime)
    
        t = ((event_t - starttime) // unit_time).astype(np.int32)
        t = np.clip(t, 0, timesteps - 1)
        p = event['p'].astype(np.int32)
        w = event['x'].astype(np.int32)
        h = event['y'].astype(np.int32)

        # Use np.add.at
        np.add.at(data, (t, p, h, w), 1.0)  # Ensure addition with float32
        data = np.clip(data, 0.0, 1.0)  # Ensure float32
        return data
    except Exception as e:
        print(e)
        return None
    
def viz_events(events,inv=False):
    events = np.load(events,allow_pickle=True)
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    if inv:
        img[events['y'], events['x']] = 255 - 255 * events['p'][:, None]
    else:
        img[events['y'], events['x']] = 255 * events['p'][:, None]
    return img
# define trining Loop

def train_loop(dataloader, model, loss_fn, optimizer,device='cuda'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    correct = 0

    scaler = GradScaler()  # Initialize GradScaler for AMP

    preprocess_times = []
    train_times = []

    model.train()  # Set model to training mode
    start_preprocess = time.time()
    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training Progress", leave=False)):
        # Start timing data preprocessing
        

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        #print(f"Input data type: {X.dtype}") 

        # End timing data preprocessing
        end_preprocess = time.time()
        
        preprocess_times.append(end_preprocess - start_preprocess)

        # Zero gradients
        optimizer.zero_grad()

        # Start timing training process
        start_train = time.time()

        with autocast(device_type="cuda"):  # Enable autocasting
            pred, _ = model(X)
            loss = loss_fn(pred, y)

        # Backpropagation with scaled loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # End timing training process
        end_train = time.time()
        train_times.append(end_train - start_train)

        # Accumulate metrics
        train_loss += loss.item()
        if pred.ndim == 3:
            correct += SF.accuracy_rate(pred, y)*y.shape[0]
        else:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        start_preprocess = time.time()

    # Calculate average loss and accuracy
    train_loss /= num_batches
    correct /= size

    # Calculate total times
    total_preprocess_time = sum(preprocess_times)/len(preprocess_times)
    total_train_time = sum(train_times)/len(train_times)

    print(f"Train Error:")
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")
    print(f"Total Preprocess Time: {total_preprocess_time:.4f}s")
    print(f"Total Training Time: {total_train_time:.4f}s")

    return {
        'loss': train_loss,
        'accuracy': correct,
        'preprocess_times': preprocess_times,
        'train_times': train_times,
    }
# define testing Loop
def test_loop(dataloader, model, loss_fn, device='cuda'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0
    num_classes = dataloader.dataset.dataset.num_classes()

    all_preds = []
    all_labels = []
    all_features = []
    all_probs = []

    inference_times = []
    preprocess_times = []

    model.eval()  # Set model to evaluation mode
    start_preprocess = time.time()
    with torch.no_grad():
        post_process = 0
        for batch, (X, y) in enumerate(tqdm(dataloader, desc="Testing Progress", leave=False)):
            # Start timing data preprocessing
            

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # End timing data preprocessing
            end_preprocess = time.time()
            preprocess_times.append(end_preprocess - start_preprocess)

            # Start timing inference
            start_inference = time.time()

            with autocast(device_type="cuda"):
                pred, features = model(X)

            # End timing inference
            end_inference = time.time()
            inference_times.append(end_inference - start_inference)

            # Compute loss and accuracy
            post_process += end_inference - start_preprocess
            
            test_loss += loss_fn(pred, y).item()
            if pred.ndim == 3:
                correct += SF.accuracy_rate(pred, y) * y.shape[0]
                features = pred.transpose(0,1)
                probs = torch.softmax(torch.sum(pred,dim=0), dim=1)
                pred_labels = torch.sum(pred, dim=0)
                pred_labels = pred_labels.argmax(1)
            else:
                probs = torch.softmax(pred, dim=1)
                pred_labels = pred.argmax(1)
                correct += (pred_labels == y).type(torch.float).sum().item()

            # Collect predictions and features
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            start_preprocess = time.time()
            

    # Calculate average loss and accuracy
    test_loss /= num_batches
    correct /= size

    # Generate classification report
    report = classification_report(all_labels, all_preds, labels=np.arange(num_classes), output_dict=True)

    # Calculate total times
    total_preprocess_time = sum(preprocess_times)/len(preprocess_times)
    total_inference_time = sum(inference_times)/len(inference_times)
    post_process /= len(dataloader)

    print(f"Test Error:")
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Total Preprocess Time: {total_preprocess_time:.4f}s")
    print(f"Total Inference Time: {total_inference_time:.4f}s")
    print(f"Post Process Time: {post_process:.4f}s")
    print("Classification Report:")
    print(report)

    return {
        'loss': test_loss,
        'accuracy': correct,
        'features': np.array(all_features),
        'labels': np.array(all_labels),
        'report': report,
        'inference_times': inference_times,
        'preprocess_times': preprocess_times,
        'probs': np.array(all_probs),
        'preds': np.array(all_preds),
    }
    
def plot_precision_recall_curves(y_true, predicted_scores, class_labels, pdf):
    # One-hot encode true labels
    num_classes = len(class_labels)
    y_true_one_hot = np.eye(num_classes)[y_true]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Loop through each class and compute Precision-Recall curve
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], predicted_scores[:, i])
        avg_precision = average_precision_score(y_true_one_hot[:, i], predicted_scores[:, i])

        # Smooth the precision-recall curve by interpolation
        recall_interp = np.linspace(recall.min(), recall.max(), 1000)  # 500 points for smooth curve
        precision_interp_func = interp1d(recall, precision, kind='linear')  # Linear interpolation
        precision_interp = precision_interp_func(recall_interp)

        # Plot smoothed Precision-Recall curve
        ax.plot(recall_interp, precision_interp, lw=2, label=f'{label} (AP = {avg_precision:.2f})')

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Smoothed Precision-Recall Curve for Each Class', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True)

    # Save the plot to the PDF
    pdf.savefig(fig)
    plt.close(fig)
    
    

def plot_results(train_result, test_result,lr_list,path):
    # for visualization during training and testing
    os.makedirs(path, exist_ok=True)

    train_loss = [result['loss'] for result in train_result]
    train_acc = [result['accuracy'] for result in train_result]
    test_loss = [result['loss'] for result in test_result]
    test_acc = [result['accuracy'] for result in test_result]
    #train_report = [result['report'] for result in train_result]
    test_report = [result['report'] for result in test_result]
    

    # 提取最后一个测试结果的特征和标签以用于UMAP可视化
    features = test_result[-1]['features']
    labels = test_result[-1]['labels']
    
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for report in test_report:
        f1_scores.append(report['macro avg']['f1-score'])
        recall_scores.append(report['macro avg']['recall'])
        precision_scores.append(report['macro avg']['precision'])
    
    

    # UMAP
    global gseed
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey']
    class_labels = ['RBC', 'Neutrophils', 'PBMC', 'Platelet', 'HUVEC', 'Microparticle']
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=gseed, n_jobs=1)
    if len(features.shape) >2:
        features = np.sum(features, axis=1)#b,t,f -b f
    embedding = reducer.fit_transform(features)

    # 绘图
    _, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs[0][0].plot(train_loss, label='Train Loss')
    axs[0][0].plot(test_loss, label='Test Loss')
    axs[0][0].set_title("Loss over Epochs")
    axs[0][0].legend()

    axs[1][0].plot(train_acc, label='Train Accuracy')
    axs[1][0].plot(test_acc, label='Test Accuracy')
    axs[1][0].set_title("Accuracy over Epochs")
    axs[1][0].legend()
    axs[0][1].plot(lr_list, label='Learning Rate')
    axs[0][1].set_title("Learning Rate over Epochs")
    axs[0][1].legend()

    # UMAP Visualization
    # mapping colors to labels
    label_to_color = {label: color for label, color in zip(range(len(class_labels)), colors)}
    point_colors = [label_to_color[label] for label in labels]
    scatter = axs[1][1].scatter(embedding[:, 0], embedding[:, 1], c=point_colors, s=5)
    axs[1][1].set_title("UMAP projection of the Dataset")
    axs[1][1].set_xlabel("UMAP Dimension 1", fontsize=14)
    axs[1][1].set_ylabel("UMAP Dimension 2", fontsize=14)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for color, label in zip(colors, class_labels)]
    plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(class_labels),
               fontsize=10)
    plt.colorbar(scatter, ax=axs[1][1], orientation='vertical')
    # save image 
    plt.savefig(f'{path}/training_results.png', dpi=300)  # 指定路径和分辨率
    plt.close()  
    _, axs = plt.subplots(1, 3, figsize=(18, 12))
    # F1 Score
    axs[0].plot(f1_scores, label='F1 Score')
    axs[0].set_title("F1 Score over Epochs")
    axs[0].legend()

    #Recall
    axs[1].plot(recall_scores, label='Recall')
    axs[1].set_title("Recall over Epochs")
    axs[1].legend()

    # Precision
    axs[2].plot(precision_scores, label='Precision')
    axs[2].set_title("Precision over Epochs")
    axs[2].legend()
    plt.savefig(f'{path}/training_eva_results.png', dpi=300)
    plt.close()  


# config set
def create_config_new(num,snnF,time_steps=4,v2f=True,start_dim=4,RlayerNum = 1, act = None,CNNType = None,SNNType = "O_Res18"):
    config_set = []
    if snnF:
        start_dim = start_dim
        if SNNType == "O_Res18":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlockVCST,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5
            (TCFATestBlockVCST,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCST,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCST,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCST,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18":
            config_set =[
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.SiLU), [0],"concat"),#3
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVCSTPN,( start_dim * 4, 3, 1, 1,time_steps, nn.SiLU), [3],"concat"),#6
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCSTPN,( start_dim * 8, 3, 1, 1,time_steps, nn.SiLU), [6], "concat"),#9
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 16, 3, 1, 1,time_steps, nn.SiLU), [9], "concat"),#12
            
            (SNNBlockVS2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 32, 3, 1, 1,time_steps, nn.SiLU), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18_2":
            config_set =[
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.SiLU), [0],"concat"),#3
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.SiLU), [3],"concat"),#6
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCSTPN,( start_dim * 4, 3, 1, 1,time_steps, nn.SiLU), [6], "concat"),#9
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 8, 3, 1, 1,time_steps, nn.SiLU), [9], "concat"),#12
            
            (SNNBlockVS2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 16, 3, 1, 1,time_steps, nn.SiLU), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18_18":
            config_set =[
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.SiLU), [0],"concat"),#2
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#3
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.SiLU), [2],"concat"),#4
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [3], "add"),#5
            (TCFATestBlockVCSTPN,( start_dim * 4, 3, 1, 1,time_steps, nn.SiLU), [4], "concat"),#6
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [5], "add"),#7
            (TCFATestBlockVCSTPN,( start_dim * 8, 3, 1, 1,time_steps, nn.SiLU), [6], "concat"),#8
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18_Nota_18":
            config_set =[
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,(start_dim * 2, time_steps,True,1,nn.SiLU), [0],"concat"),#2
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#3
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [2],"concat"),#4
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [3], "add"),#5
            (SnnResidualBlock,( start_dim * 4,time_steps,True,1, nn.SiLU), [4], "concat"),#6
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [5], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [6], "concat"),#8
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18_Nota_Nosk_18":
            config_set =[
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,(start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [], None),#3
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [], None),#4
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [], None),#5
            (SnnResidualBlock,( start_dim * 4,time_steps,True,1, nn.SiLU),[], None),#6
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [], None),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Res18_ATP":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlockVCSTP,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVCSTP,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCSTP,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTP,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTP,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Res18_ATPN":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [0],"concat"),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVCSTPN,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCSTPN,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCSTPN,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Res18_NoTA":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [0],"concat"),#2
            #(TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [3],"concat"),
            #(TCFATestBlock,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (SnnResidualBlock,( start_dim * 4, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            #(TCFATestBlock,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 8, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            #(TCFATestBlock,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1,nn.SiLU), [3],"concat"),#2
            #(TCFATestBlock,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,nn.SiLU,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,nn.SiLU,True), [], None),
        ]
        elif SNNType == "O_Strict_Res18_NoTA":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [0],"concat"),#2
            #(TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [3],"concat"),
            #(TCFATestBlock,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (SnnResidualBlock,( start_dim * 4, time_steps,True,1,nn.SiLU), [6],"concat"),#2
            #(TCFATestBlock,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 8, time_steps,True,1,nn.SiLU), [9],"concat"),#2
            #(TCFATestBlock,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockVS2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1,nn.SiLU), [12],"concat"),#2
            #(TCFATestBlock,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,nn.SiLU,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,nn.SiLU,True), [], None),
        ]
        elif SNNType == 'O_Strict_Res18_NoTA_NoSk':
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockVS2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            #(TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockVS2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [], None ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [], None),
            #(TCFATestBlock,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"conct"),#6
            
            (SNNBlockVS2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [], None),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (SnnResidualBlock,( start_dim * 4, time_steps,True,1,nn.SiLU), [], None),#2
            #(TCFATestBlock,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockVS2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [], None),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 8, time_steps,True,1,nn.SiLU), [], None),#2
            #(TCFATestBlock,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockVS2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [], None), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1,nn.SiLU), [], None),#2
            #(TCFATestBlock,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,nn.SiLU,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,nn.SiLU,True), [], None),
        ]
        elif SNNType == "O_Res18_CST":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVCST,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVCST,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCST,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVCST,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "O_Res18_TCS":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (SNNBlockV3M2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,None), [], None),#2
            (TCFATestBlockVTCS,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (SNNBlockV3M2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlockVTCS,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (SNNBlockV3M2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlockVTCS,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (SNNBlockV3M2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVTCS,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (SNNBlockV3M2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlockVTCS,( start_dim * 32, 3, 1, 1,time_steps, nn.ReLU6), [12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType =="MS_Res18":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (MSM2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [],None),#3
            
            (MSM2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (SnnResidualBlock,( start_dim * 2,  time_steps,True,1,nn.SiLU), [3],"concat"),#6
            
            (MSM2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (SnnResidualBlock,( start_dim * 4,time_steps,True,1, nn.SiLU), [6], "concat"),#9
            
            (MSM2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 8, time_steps,True,1, nn.SiLU), [9], "concat"),#12
            
            (MSM2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU),[12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
        elif SNNType == "MS_Res18_no_TA":
            config_set = [
            (TCnnbase, ( start_dim,7,1,3,time_steps, snn.BatchNormTT2d, None), [], None),#0
            
            (MSM2, ( start_dim * 2,  3, 2, 1,time_steps, v2f, act), [], None),#1
            (SnnResidualBlock,( start_dim * 2, time_steps,True,1,nn.SiLU), [], None),#2
            (TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [],None),#3
            
            (MSM2, ( start_dim * 4, 3, 2, 1,time_steps, v2f, act), [1],"add" ),#4
            (SnnResidualBlock,( start_dim * 4,  time_steps,True,1,nn.SiLU), [], None),#5 out_channels, time_steps,use_residual=True,num_repeats = 1,activation = nn.SiLU
            (TCFATestBlock,( start_dim * 2, 3, 1, 1,time_steps, nn.ReLU6), [3],"concat"),#6
            
            (MSM2, ( start_dim * 8,3, 2, 1,time_steps, v2f, act), [4], "add"),#7
            (SnnResidualBlock,( start_dim * 8,time_steps,True,1, nn.SiLU), [], None),#8
            (TCFATestBlock,( start_dim * 4, 3, 1, 1,time_steps, nn.ReLU6), [6], "concat"),#9
            
            (MSM2, ( start_dim * 16, 3, 2, 1,time_steps, v2f, act), [7], "add"),#10
            (SnnResidualBlock,( start_dim * 16, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlock,( start_dim * 8, 3, 1, 1,time_steps, nn.ReLU6), [9], "concat"),#12
            
            (MSM2, ( start_dim * 32, 3, 2, 1,time_steps, v2f, act), [10], "add"), #13
            (SnnResidualBlock,( start_dim * 32, time_steps,True,1, nn.SiLU), [], None),#11
            (TCFATestBlock,( start_dim * 16, 3, 1, 1,time_steps, nn.ReLU6),[12], "concat"),#14
            
            (SnnMLPLayer,(1000,time_steps,snn.BatchNormTT1d,None,False), [], None),
            (SnnMLPLayer,(num,time_steps,snn.BatchNormTT1d,None,True), [], None),
        ]
    else:
        if CNNType is None:
            start_dim = start_dim
            config_set=[
                (Cnnbase, (start_dim*2, 3, 1, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (Cnnbase, (start_dim*4, 3, 2, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (ResidualBlock, (start_dim*4, 1,None,2), [], None),

                (Cnnbase, (start_dim*8, 3, 2, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (ResidualBlock, (start_dim*8, 1,None,4), [], None),

                (Cnnbase, (start_dim*16, 3, 2, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (ResidualBlock, (start_dim*16, 1,None,8), [], None),
                (Cnnbase, (start_dim*32, 3, 2, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (ResidualBlock, (start_dim*32, 1,None,8), [], None),
                (Cnnbase, (start_dim*64, 3, 2, 1, nn.BatchNorm2d,nn.ReLU6), [], None),  # This layer's output will be used later
                (nn.Linear, (1000,), [], None),
            (nn.Linear, (num,), [], None)]
        elif CNNType == "DeelpCnn8":
            config_set = [
                "DeelpCnn8"
            ]
        elif CNNType == "Deelpflow":
            config_set = [
                "Deelpflow"
            ]
        elif CNNType == "ResNet18":
            config_set = [
             "ResNet18",   
            ]
        elif CNNType == "ResNet34":
            config_set = [
             "ResNet34",  
            ]
        elif CNNType == "ResNet50":
            config_set = [
             "ResNet50",  
            ]
        elif CNNType == "ResNet101":
            config_set = [
             "ResNet101",   
            ]
        elif CNNType == "ResNet152":
            config_set = [
             "ResNet152",   
            ]
        elif CNNType == "Inception_v3":
            config_set = [
             "Inception_v3",   
            ]
    return config_set

# dataset class
class CustomImageDataset_imF2(Dataset):
    def __init__(self, root_dir, snnF=False, transform=None, cache_dir=None, 
                 preload=False, balance_data=True, sample_size=1000, sampling_strategy='equal_samples', imbalance=None,time_stepd=4):
        """
        Args:
            root_dir (string): dataset root directory
            cache_dir (string): cache directory 
            transform (callable, optional): default is None, for inception_v3 you need one to transform size to (299, 299)
        """
        self.root_dir = Path(root_dir)  
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.snnF = snnF
        self.cache_dir = Path(cache_dir) if cache_dir else None 
        self.preload = preload
        self.preloaded_data = []
        self.sampling_strategy = sampling_strategy
        self.time_stepd = time_stepd
        
        # create labels based on sub-folders
        self._prepare_dataset(balance_data, sample_size, imbalance)
        if preload:
            self._preload_data()
    
    def _prepare_dataset(self, balance_data=True, sample_size=1000, imbalance=None):
        # Initialize class_samples dictionary
        self.samples_per_class = {class_name: 0 for class_name in self.class_to_idx.keys()}
        setted_dict = {
            'RBC': 0, 
            'Neutrophils': 1, 
            'PBMC': 2, 
            'Platelet': 3,
            'HUVEC': 4, 
            'Microparticle': 5
        }
        class_samples = defaultdict(list)
        
        for folder_name in os.listdir(self.root_dir):
            folder_path = self.root_dir / folder_name  # 使用 Path 对象构造路径
            setted_idx = setted_dict.get(folder_name, None)
            if setted_idx is not None and folder_path.is_dir():
                self.class_to_idx[folder_name] = setted_idx
                for img_name in os.listdir(folder_path):
                    img_path = folder_path / img_name  # 使用 Path 对象构造路径
                    if self.snnF and img_name.lower().endswith('.npy'):
                        class_samples[folder_name].append((img_path, setted_idx))
                    elif not self.snnF and img_name.lower().endswith('.npy'):
                        class_samples[folder_name].append((img_path, setted_idx))
                        
        if balance_data:
            if self.sampling_strategy == 'max_samples':
                for class_name, samples in class_samples.items():
                    num_samples = min(len(samples), sample_size)
                    sampled = random.sample(samples, num_samples)
                    self.image_paths.extend([s[0] for s in sampled])
                    self.labels.extend([s[1] for s in sampled])
                    self.samples_per_class[class_name] = num_samples
                    
            elif self.sampling_strategy == 'equal_samples':
                num_samples = min(len(samples) for samples in class_samples.values())
                for class_name, samples in class_samples.items():
                    sampled = random.sample(samples, num_samples)
                    self.image_paths.extend([s[0] for s in sampled])
                    self.labels.extend([s[1] for s in sampled])
                    self.samples_per_class[class_name] = num_samples
            
        else:
            # if do not want to balance data, just add all samples into dataset
            for samples in class_samples.values():
                self.image_paths.extend([s[0] for s in samples])
                self.labels.extend([s[1] for s in samples])
        
        if imbalance is not None and balance_data:
            # for imbalance test
            id, rate = imbalance
            if isinstance(id, str):
                class_name = id
            elif isinstance(id, int):
                class_name = next((name for name, idx in setted_dict.items() if idx == id), None)
            else:
                raise ValueError("target_class should be a class name (str) or class index (int).")
            
            if class_name is None or class_name not in class_samples:
                raise ValueError(f"Class '{class_name}' not found in dataset.")
            indices_to_reduce = [i for i, label in enumerate(self.labels) if label == setted_dict[class_name]]
            reduce_num = int(len(indices_to_reduce) * 5 / rate)
            indices_to_keep = random.sample(indices_to_reduce, reduce_num)
            indices_to_remove = set(indices_to_reduce) - set(indices_to_keep)
            for idx in sorted(indices_to_remove, reverse=True):
                del self.image_paths[idx]
                del self.labels[idx]
            # update counting numbers
            self.samples_per_class[class_name] = reduce_num
            
        # Output the number of samples per class
        print("Number of samples per class:")
        for class_name, count in self.samples_per_class.items():
            print(f"  Class '{class_name}' ({setted_dict[class_name]}): {count} samples")

    def _preload_data(self):
        """
        preloading all data into RAM
        """
        for img_path in self.image_paths:
            image = self._load_data(img_path)
            self.preloaded_data.append(image)
    
    def _load_data(self, eventfile):
        if self.cache_dir:
            cache_file = self.cache_dir / f"{eventfile.stem}.pkl"  
            if cache_file.exists():
                with cache_file.open('rb') as f:
                    data = pickle.load(f)
                    data = data.astype(np.float32)  # Ensure data is float32
                    return data
        if self.snnF:
            data = direct_event_to_tensor(eventfile, type='npy', timesteps=self.time_stepd)
        else:
            data = viz_events(eventfile)
        data = data.astype(np.float32)  # Ensure data is float32
        if self.cache_dir:
            with cache_file.open('wb') as f:
                pickle.dump(data, f)
        return data

    def _clear_cache(self):
        # clean all cached file 
        if self.cache_dir:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(self.cache_dir / filename)  
            print('cache cleared')
                
    def __len__(self):
        # return the total number of samples
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.preload:
            image = self.preloaded_data[idx]
        else:
            img_path = self.image_paths[idx]
            image = self._load_data(img_path)
        
        if self.transform:
            if self.snnF:
                for t in range(image.shape[0]):
                    image[t] = self.transform(image[t])
                image = torch.stack(image)
            else:
                image = transforms.ToTensor()(image)
                image = self.transform(image)
        else:
            if self.snnF:
                image = torch.from_numpy(image)
            else:
                image = transforms.ToTensor()(image)
        label = self.labels[idx]
        
        return image, label

    def num_classes(self):
        return len(self.class_to_idx)

    def save_class_indices(self, filename='class_indices.txt'):
        with open(filename, 'w') as file:
            for class_name, idx in sorted(self.class_to_idx.items()):
                file.write(f'{idx}: {class_name}\n')            
                
                
    

# import tensorflow_datasets as tfds
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def trainer(path,cache_path,name,transform,lr,in_channels=4,start_dim=4,snnF=False,v2f=True,sample_size=1000,rnum=1,act =None,ep=300,CNNType = None,SNNType = "O_Res18",imbalance = None,time_step=4):
    base_name = Path.cwd() / f'run_{name}'
    new_name = base_name
    # create a new name if it already exists
    counter = 1
    while new_name.exists():
        new_name = Path(f'{base_name}_{counter}')
        counter += 1
    
    name = new_name
    name.mkdir(parents=True, exist_ok=False)
    global gseed
    g = torch.Generator()
    g.manual_seed(gseed)
    dataset = CustomImageDataset_imF2(root_dir=path,snnF=snnF,transform=transform,
                                 cache_dir=cache_path,balance_data=True,preload=False,sample_size=sample_size,imbalance=imbalance,time_stepd=time_step)
    numberclass = dataset.num_classes()
    conf = create_config_new(numberclass,snnF,time_step,v2f,start_dim,rnum,act,CNNType,SNNType)
    print(conf)
    if CNNType is not None and snnF is False:
        print(f"Using {CNNType} CNN")
        if CNNType == "DeelpCnn8":
            model = MD.DeelpCnn8(in_channels,numberclass) #model = MD.BaseLineModel(conf,in_channels,numberclass)
        elif CNNType == "Deelpflow":
            model = MD.DeepflowModel(in_channels,numberclass)
        elif CNNType == "InceptionV3":
            model = MD.CustomInceptionV3(in_channels,numberclass)
        else:
            model = MD.BaseLineModel(conf,in_channels,numberclass)
    else:
        model = MD.ClassificationModel_New_New(numberclass,in_channels,conf,snnF,timestpes=time_step)
    
    model = model.to('cuda')
    model_conf_path = name / "model_conf.pkl"
    with open(model_conf_path, 'wb') as f:
        pickle.dump(conf, f)
    
    dataset.save_class_indices(filename= name/'class_indices.txt')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4,pin_memory=True,generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True,num_workers=4,pin_memory=True,generator=g)
    if snnF:
        loss_fn =SF.ce_rate_loss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,betas=(0.9,0.99))
    if time_step != 1:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2, verbose=True,threshold=1e-4,threshold_mode='abs',cooldown=0,min_lr=0.00005,eps=1e-8)
    else: 
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2, verbose=True,threshold=1e-4,threshold_mode='abs',cooldown=0,min_lr=0.00005,eps=1e-8)
    training_result_list = []
    testing_result_list = []
    lr_list = []
    best_epoch = -1
    best_f1_score = 0
    epochs = ep
    for ep in tqdm(range(epochs), desc="Training Progress"):
        training_result_list.append(train_loop(train_dataloader, model, loss_fn, optimizer)) 
        testing_result_list.append(test_loop(test_dataloader, model, loss_fn))
        val_loss = testing_result_list[-1]['loss']
        if isinstance(scheduler,lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        test_report = testing_result_list[-1]['report']
        f1_score = test_report['weighted avg']['f1-score']

        # save the best model
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            if best_epoch > 0:
                os.remove(os.path.join(name, f"best_model_epoch_{best_epoch}.pth"))
            best_epoch =  + 1  
            model_save_path = os.path.join(
                                name, f"best_model_epoch_{best_epoch}.pth"
                            )
            torch.save(model, model_save_path)

            print(f"New best model saved with F1 Score: {best_f1_score:.4f} at epoch {best_epoch}")
                        
        if f1_score >0.93:
            # plot out all the results meeting the threshold
            name_2 = name / f"fl_{f1_score:.4f}_epoch_{ep}"
            if not name_2.exists():
                name_2.mkdir(parents=True, exist_ok=False)
            plot_results(training_result_list, testing_result_list, lr_list,name_2)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {ep+1}/{epochs}, Current Learning Rate: {current_lr}")
        lr_list.append(current_lr)
        print(f"Train Loss: {training_result_list[-1]['loss']:.4f}, Train Accuracy: {training_result_list[-1]['accuracy']:.4f}")
        print(f"Test Loss: {testing_result_list[-1]['loss']:.4f}, Test Accuracy: {testing_result_list[-1]['accuracy']:.4f}")
        if ep % 10 == 0:
            plot_results(training_result_list, testing_result_list, lr_list,name)
            
            np.save(name/"training_results.npy", training_result_list)
            np.save(name/"testing_results.npy", testing_result_list)
            np.save(name/"lr_list.npy", lr_list)

    plot_results(training_result_list, testing_result_list, lr_list, name)

    torch.save(model.state_dict(), name/"model.pth")
    np.save(name/"training_results.npy", training_result_list)
    np.save(name/"testing_results.npy", testing_result_list)
    np.save(name/"lr_list.npy", lr_list)
    
    dataset._clear_cache()
    return training_result_list, testing_result_list,name


if __name__ == "__main__":
    #test_training_pipeline()
    
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    set_seed(gseed)
    '''
    # for CNN model
    lr = 0.0001
    samples = 5000
    start_dim = 4
    rnum = 1
    snnf = False
    v2f = False
    SNNtype = None
    time_step =1
    input_dim = 3
    transform_inc = None#transforms.Compose([
    #transforms.Resize((299, 299)),])
    for i in [None]:
        for v2f in [False]:
            for CNNtype in ["ResNet18"]:
                trainer("/home/hx/Documents/ai/Object_detect/data/train_Rename_arg_r",'/media/hx/J/cache6',f"test_snn_cachetest_32_47_geed_R_Cnn_v2_noFr_output_{snnf}_{lr}_{start_dim}_ATT_{v2f}_{samples}_repeatnum{rnum}_SNN_{SNNtype}_CNN_{CNNtype}_imf_{i}_timestep_{time_step}"
                ,transform_inc         
                ,lr,input_dim,start_dim,snnf,v2f,samples,rnum,None,100,CNNtype,SNNtype,imbalance = i,time_step = time_step)
    '''
    #for SNN model
    lr = 0.0008
    samples = 5000
    start_dim = 4
    rnum = 1
    snnf = True
    v2f = False
    CNNtype = None
    DataPath = "/home/hx/Documents/ai/Object_detect/data/train_Rename_arg_r"
    CachePath = "/media/hx/J/cache6"
    
    
    input_dim = 2
    for  time_step in [1]:
        for i in [None]:
            for v2f in [False]:
                for SNNtype in ["O_Strict_Res18_18"]:#"O_Strict_Res18","O_Res18_ATPN"
                    if time_step == 4:
                        lr = 0.0001
                    Name = f"test_snn_cachetest_32_47_full_geed_R_Snn_v2_noFr_output_{snnf}_{lr}_{start_dim}_ATT_{v2f}_{samples}_repeatnum{rnum}_SNN_{SNNtype}_imf_{i}_timestep_{time_step}"
                    trainer(DataPath,CachePath
                            ,Name
                            ,None,lr,input_dim,start_dim,snnf,v2f,samples,rnum,None,100,CNNtype,SNNtype,imbalance = i,time_step = time_step)
    lr = 0.0008
    samples = 5000
    start_dim = 4
    rnum = 1
    snnf = True
    v2f = False
    CNNtype = None
    DataPath = "/home/hx/Documents/ai/Object_detect/data/train_Rename_arg_r"
    CachePath = "/media/hx/J/cache6"
    input_dim = 2
    for  time_step in [1]:
        for i in [(0,50),(1,50),(2,50),(3,50),(4,50),(5,50)]:
            for v2f in [False]:
                for SNNtype in ["O_Strict_Res18_18"]:#"O_Strict_Res18","O_Res18_ATPN"
                    if time_step == 4:
                        lr = 0.0001
                    Name = f"test_snn_cachetest_32_47_full_geed_R_Snn_v2_noFr_output_{snnf}_{lr}_{start_dim}_ATT_{v2f}_{samples}_repeatnum{rnum}_SNN_{SNNtype}_imf_{i}_timestep_{time_step}"
                    trainer(DataPath
                            ,CachePath
                            ,Name,None,lr,input_dim,start_dim,snnf,v2f,samples,rnum,None,100,CNNtype,SNNtype,imbalance = i,time_step = time_step)
    