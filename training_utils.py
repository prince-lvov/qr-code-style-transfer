import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

from models import QuantAE, QuantAEPruned


LATENT_DIMS = {
    41: 882,
    21: 220
}

ENCODER_DIMS = {
   41: [16, 32, 64],
   21: [16, 32, 64]
}

DECODER_DIMS = {
   41: [1681, 5043],
   21: [441, 1323]
}

ENCODER_OUT_CHANNEL_SIZES = {
   41:  { 16: (21, 21), 32: (11, 11), 64: (6, 6) },
   21:  { 16: (11, 11), 32: (6, 6), 64: (3, 3) },
}


class QRCodeImageDataset(Dataset):
    def __init__(self, images_np, images_styled_np, image_size):
        self.images_np = images_np
        self.images_styled_np = images_styled_np
               
        # Grayscalled: .299 Red + 0.587 Green + 0.114 Blue => mean = 0.458971, std = 0.225609
        self.transform_input = transforms.Compose([
            transforms.CenterCrop(image_size)
          ])
        
        self.transform_styled = transforms.Compose([
            transforms.CenterCrop(image_size)
          ])

    def __len__(self):
        return self.images_styled_np.shape[0]

    def __getitem__(self, idx):
        image_np = self.images_np[idx]
        image_styled_np = self.images_styled_np[idx]
        
        image = torch.from_numpy(image_np).float().unsqueeze(0)
        image_styled = torch.from_numpy(np.transpose(image_styled_np,(2,0,1))).float() / 255

        return image, image_styled


def create_qrcodes_datasets(dataset_dir, dataset_size):
    dataset = []
    for i in range(1, dataset_size + 1):
        image = Image.open(f"{dataset_dir}/{i}.jpg")
        dataset.append(np.asarray(image))
    return np.split(dataset, [int(.8 * len(dataset)), int(.95 * len(dataset))])


def get_ae_model(name, image_size, debug=False):
	return QuantAE(
        name=name,
		in_channels=1,
		latent_dim=LATENT_DIMS[image_size],
        encoder_dims=ENCODER_DIMS[image_size],
        decoder_dims=DECODER_DIMS[image_size],
        encoder_out_channel_sizes=ENCODER_OUT_CHANNEL_SIZES[image_size],
		image_size=image_size,
        debug=debug   
	)
    
def get_ae_pruned_model(name, image_size, debug=False):
   return QuantAEPruned(
        name=name,
        in_channels=1,
        latent_dim=LATENT_DIMS[image_size],
        encoder_dims=ENCODER_DIMS[image_size],
        decoder_dims=DECODER_DIMS[image_size],
        encoder_out_channel_sizes=ENCODER_OUT_CHANNEL_SIZES[image_size],
		image_size=image_size,
        debug=debug   
	)


def load_model(model_name, image_size, pruned=False):
    if pruned:
        model = get_ae_pruned_model(model_name, image_size)
    else:
        model = get_ae_model(model_name, image_size)
    model_checkpoint_name = f'{model_name}/final_ckpt.pt'
    model.load_state_dict(
        torch.load(model_checkpoint_name)
    )
    print(f"Model state {model_checkpoint_name} was loaded")
    return model


def save_model(model):
    torch.save(model.state_dict(), f"{model.name}/final_ckpt.pt")


def save_compiled_model(fhe_model):
    fhe_model.fhe_circuit.client.save("./client.zip")
    fhe_model.fhe_circuit.server.save("./server.zip")


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').detach().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def predict_images(model: nn.Module, input: torch.Tensor,
                   current_device: int = 0, **kwargs) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
      predict = model(input)
    
    model.train()
    predict = predict.clamp(0, 1)
    predict = (predict * 255).type(torch.uint8)

    return predict


def save_train_results(model_name, start_epoch, epochs, train_results):
    plt.plot(train_results["train_loss"], label="training loss")
    plt.plot(train_results["val_loss"], label="val loss")
    plt.legend(loc="upper right")

    loss_image_name = f"{model_name}/loss_{start_epoch}_{epochs}.png"
    plt.savefig(loss_image_name)
    plt.show()


def train(model, dataloader_train, dataloader_val, stage, params):
    work_dir = Path(model.name)
    img_dir = work_dir / "img"
    states_dir = work_dir / "states"
    work_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    states_dir.mkdir(exist_ok=True)

    train_results = {
       "train_loss": [],
       "val_loss": []
    }

    device = params["device"]        
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['LR'], weight_decay=params['weight_decay']
    )
    lr_last = params['LR']
    loss_fn = F.l1_loss # F.mse_loss
    
    scheduler = None
    if params['scheduler_gamma'] is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = params['scheduler_gamma'])

    total_iterations = 0
    start_epoch = params["stages"][stage]["start_epoch"]
    epochs = params["stages"][stage]["epochs"]
    for epoch in range(start_epoch, epochs, 1):
        if device == "cuda":
            torch.cuda.empty_cache()

        model.train().requires_grad_(True).to(device=device)

        total = None           
        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(dataloader_train, desc=f'[Training] Epoch {epoch+1}', total=total)
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            
            image, image_styled = data
            image = image.to(device)
            image_styled = image_styled.to(device)
            
            optimizer.zero_grad()
            image_predicted = model(image)
            loss = loss_fn(image_predicted, image_styled)*255
            loss_value = loss.item()
          
            # Correct learning rate and repeat forward
            correction_cnt = 0
            while np.isnan(loss_value):
                correction_cnt += 1
                if correction_cnt > 10:
                    print("Learning process was broken due to excess of LR correction count")
                    return

                lr_corrected = 5*lr_last
                print(f"Learning rate was CORRECTED, new lr: {lr_corrected}")
                for g in optimizer.param_groups:
                    g['lr'] = lr_corrected
                lr_last = lr_corrected
                
                optimizer.zero_grad()
                image_predicted = model(image)
                loss = loss_fn(image_predicted, image_styled)*255
                loss_value = loss.item()

            loss_sum += loss_value
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(avg_loss=avg_loss)
          
            loss.backward()
            optimizer.step()            
            
      
        train_results["train_loss"].append(loss_sum / num_iterations)

        # Validation:
        torch.cuda.empty_cache()
        model.eval()
        loss_sum_val = 0
        num_iterations_val = 0
        data_iterator = tqdm(dataloader_val, desc=f'[Validation] Epoch {epoch+1}', total=total)
        for data in data_iterator:
            num_iterations_val += 1
            
            image, image_styled = data
            image = image.to(device)
            image_styled = image_styled.to(device)
            
            image_predicted = model(image)
            loss = loss_fn(image_predicted, image_styled)*255
            loss_value = loss.item()
                    
            loss_sum_val += loss_value
            avg_loss = loss_sum_val / num_iterations_val
            data_iterator.set_postfix(avg_loss=avg_loss)
      
        train_results["val_loss"].append(loss_sum_val / num_iterations_val)
      
        if scheduler:
            scheduler.step()
            lr_last = scheduler.get_last_lr()
            print(f"   > scheduler next lr: {scheduler.get_last_lr()}")

        if (epoch+1) %10 == 0:
            image_predicted_rnd = predict_images(model, image)[random.randint(0, image.shape[0]-1)]
            print(f"rundom predicted image {epoch+1}.jpg ({image_predicted_rnd.shape}, {image_predicted_rnd.dtype})")
            save_images(image_predicted_rnd, img_dir / f"{epoch+1}.jpg")
      
        if (epoch+1) %30 == 0:
            torch.save(model.state_dict(), states_dir / f"ckpt_{epoch+1}.pt")
            print(f"model state saved: ckpt_{epoch+1}.pt")


    save_train_results(model.name, start_epoch, epochs, train_results)