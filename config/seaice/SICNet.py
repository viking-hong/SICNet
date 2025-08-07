from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.seaice_dataset import *
from geoseg.models.SICNet import SICNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils

net_name = 'SICNet'
cutmix = False

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 12
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = net_name
weights_path = "model_weights/seaice/{}".format(weights_name)
test_weights_name = net_name
log_name = 'seaice/{}'.format(weights_name)
monitor = 'val_FWIOU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader

train_dataset = SeaIceDataset(data_root='D:/Dataset-train/CDbohai-D/train', mode='train',
                                 mosaic_ratio=0.25, mixup=cutmix)

val_dataset = SeaIceDataset(data_root='D:/Dataset-train/CDbohai-D/test')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)
#############
if net_name == 'SICNet':
    net = SICNet(input_nbr=3, label_nbr=num_classes)
else:
    print("未定义模块")
##############

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

