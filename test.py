import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.visual_front import Visual_front
from src.models.audio_front import Audio_front
from src.models.memory import Memory
from src.models.temporal_classifier import Temp_classifier
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.vid_aud_lrw import MultiDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import torch.nn.parallel
import math
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrw', default="Data_dir")
    parser.add_argument('--model', default="Resnet18")
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/LRW_18_mstcn_MHVAM')
    parser.add_argument("--checkpoint", type=str, default='Checkpoint_dir')
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--slot", type=int, default=112)
    parser.add_argument("--head", type=int, default=8)

    parser.add_argument("--max_timesteps", type=int, default=29)
    parser.add_argument("--test_aug", default=False, action='store_true')

    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = '5555'

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    v_front = Visual_front(in_channels=1)
    a_front = Audio_front(in_channels=1)
    mem = Memory(radius=args.radius, n_slot=args.slot, n_head=args.head)
    tcn = Temp_classifier(radius=args.radius, n_slot=args.slot, head=args.head)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        a_front.load_state_dict(checkpoint['a_front_state_dict'])
        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        mem.load_state_dict(checkpoint['mem_state_dict'])
        tcn.load_state_dict(checkpoint['tcn_state_dict'])
        del checkpoint

    v_front.cuda()
    a_front.cuda()
    mem.cuda()
    tcn.cuda()

    if args.distributed:
        v_front = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v_front)
        a_front = torch.nn.SyncBatchNorm.convert_sync_batchnorm(a_front)
        mem = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mem)
        tcn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcn)

    if args.distributed:
        v_front = DDP(v_front, device_ids=[args.local_rank], output_device=args.local_rank)
        a_front = DDP(a_front, device_ids=[args.local_rank], output_device=args.local_rank)
        mem = DDP(mem, device_ids=[args.local_rank], output_device=args.local_rank)
        tcn = DDP(tcn, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataparallel:
        v_front = DP(v_front)
        a_front = DP(a_front)
        mem = DP(mem)
        tcn = DP(tcn)

    _ = test(v_front, a_front, mem, tcn)

def test(v_front, a_front, mem, tcn, fast_validate=False):
    with torch.no_grad():
        a_front.eval()
        v_front.eval()
        mem.eval()
        tcn.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        criterion = nn.CrossEntropyLoss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        tot_cor, tot_v_cor, tot_a_cor, tot_num = 0, 0, 0, 0

        description = 'Check test step' if fast_validate else 'Test'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            a_in, v_in, target = batch

            v_feat = v_front(v_in.cuda())   #B,S,512
            a_feat = a_front(a_in.cuda())   #B,S,512
            te_fusion, _, _, _ = mem(v_feat, a_feat, inference=True)
            te_m_pred, _, _ = tcn(te_fusion, None, infer=True, mode='te')

            ori_pred = te_m_pred.clone().cpu()
            ################## flip ##################
            if args.test_aug:
                v_in = v_in.flip(4)  # B,C,T,H,W --> B,C,T,H,W

                v_feat = v_front(v_in.cuda())   #B,S,512
                a_feat = a_front(a_in.cuda())   #B,S,512
                te_fusion, _, _, _ = mem(v_feat, a_feat, inference=True)
                te_m_pred, _, _ = tcn(te_fusion, None, infer=True, mode='te')
            else:
                ori_pred = 0.

            loss = criterion(te_m_pred, target.long().cuda()).cpu().item()
            prediction = torch.argmax(te_m_pred.cpu() + ori_pred, dim=1).numpy()

            tot_cor += np.sum(prediction == target.long().numpy())

            tot_num += len(prediction)

            batch_size = te_m_pred.size(0)
            val_loss.append(loss)

            if i >= max_batches:
                break

        a_front.train()
        v_front.train()
        mem.train()
        tcn.train()
        print('Test_ACC:', tot_cor / tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num


if __name__ == "__main__":
    args = parse_args()
    train_net(args)

