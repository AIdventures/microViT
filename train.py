"""
Training script for the ViT model.

Run:
$ python train.py

Notes:
- Requires an account at wandb.ai to log the training process.
"""
import os
import time
import wandb
import torch
import torchvision
from torch.nn import functional as F
from transformers import get_scheduler

from model import ViT, get_accuracy
from config import VIT_CONFIG


###################################################################################
################################## CONFIGURATION ##################################
###################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {k:v for k,v in VIT_CONFIG.__dict__.items() if '__' not in k}
config['max_iters'] = 1000
config['eval_interval'] = 50
config['batch_size'] = 64
config['desired_batch_size'] = 256
config['gradient_accumulation_steps'] = config['desired_batch_size'] // config['batch_size']
config['lr'] = 1e-3
config['weight_decay'] = 0.01
config['warmup_steps'] = 100
config['beta1'] = 0.9
config['beta2'] = 0.999

###################################################################################
#################################### LOGGING ######################################
###################################################################################
os.environ["WANDB_SILENT"] = "true"
wandb_project = 'microViT_mnist'
run = wandb.init(project=wandb_project, config=config)
print(f'Run name: {run.name}. Visit at {run.get_url()}')


###################################################################################
################################## DATA RELATED ###################################
###################################################################################
# Define transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Initialize datasets and dataloaders
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

# Calculate the sizes of train and validation sets
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Create indices for train and validation sets
indices = list(range(len(train_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Define samplers for train and validation sets
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# Create train and validation dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, sampler=train_sampler
)
val_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, sampler=val_sampler
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)


###################################################################################
################################## MODEL SETUP ####################################
###################################################################################
model = ViT(VIT_CONFIG)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=config['lr'],
    betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay']
)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer,
    num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_iters']
)


###################################################################################
################################## TRAINING #######################################
###################################################################################
t0 = time.time()
for iter_num in range(config['max_iters']):

    for micro_step in range(config['gradient_accumulation_steps']):
        # Extract a batch of data
        batch = next(iter(train_dataloader))

        x, y = batch[0].to(device), batch[1].to(device)

        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        # scale the loss to account for gradient accumulation
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # scale up to undo the division above
    # approximating total loss (exact would have been a sum)
    lossf = loss.item() * config['gradient_accumulation_steps']
    dt = (time.time() - t0) // 60
    print(f"iter {iter_num}: loss {lossf:.6f}, time {dt:g}min")

    # evaluation
    if iter_num % config['eval_interval'] == 0:
        val_acc = get_accuracy(model, val_dataloader, device)
        train_acc = get_accuracy(model, train_dataloader, device)
        print(f"=> train_acc {train_acc:.4f}, val_acc {val_acc:.4f}")

        wandb.log({
            "iter": iter_num,
            "train/lossf": lossf,
            "train/acc": train_acc,
            "val/acc": val_acc,
            "lr": lr_scheduler.get_last_lr()[0],
        })


###################################################################################
################################## CHECKPOINT #####################################
###################################################################################
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    "train/acc": train_acc,
    "val/acc": val_acc,
    'config': config,
}
print(f"saving checkpoint to {run.name}")
os.makedirs(os.path.join("checkpoints", run.name), exist_ok=True)
torch.save(checkpoint, os.path.join("checkpoints", run.name, 'latest_ckpt.pt'))


###################################################################################
################################## TESTING ########################################
###################################################################################
test_acc = get_accuracy(model, test_dataloader, device)
print(f"test_acc {test_acc:.4f}")
wandb.summary["test/acc"] = test_acc
wandb.finish()
print(f"Done! Visit at {run.get_url()}")
