"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, sys

import numpy as np
from PIL import Image
import torch as th
from torchvision import utils
import torch.distributed as dist

from improved_diffusion.utils import get_stain_matrix

from improved_diffusion import dist_util, logger, image_datasets
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
###

def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = image_datasets.load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond)
    for large_batch, model_kwargs, filename in data:
        model_kwargs['ref_img'] = large_batch
        yield model_kwargs, filename

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        #dist_util.load_state_dict(args.model_path, map_location="cpu")
        th.load(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond)
    args.num_samples = len(os.listdir(args.base_samples))
    assert args.num_samples >= args.batch_size * dist_util.get_world_size()
    
    # get stain matrix
    ToUint8 = lambda img: (img + 1) * 127.5
    ToNormalized = lambda img: (img / 127.5) - 1
    ToNPArray = lambda img: np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    
    stain_matrix_S = th.from_numpy(np.load(args.stain_matrix_dir).T).to(dist_util.dev())
    stain_matrix_S_inv = th.linalg.pinv(stain_matrix_S)
    
    logger.log("sampling...")
    count = 0
    while count * args.batch_size * dist_util.get_world_size() < args.num_samples:
        model_kwargs, filename = next(data)
        current_batch_size = len(filename)
        
        A = th.zeros(current_batch_size, 3, 3).to(dist_util.dev())
        Y_od = th.zeros(current_batch_size,
                        model_kwargs['ref_img'].size(dim=1),
                        model_kwargs['ref_img'].size(dim=2)*model_kwargs['ref_img'].size(dim=3)).to(dist_util.dev())
        for idx in range(current_batch_size):
            y = ToUint8(model_kwargs['ref_img'][idx,:,:,:].squeeze())
            y = th.maximum(th.minimum(y, th.tensor(255)), th.tensor(1))
            y_od = th.maximum(th.log(255. / y), th.tensor(0)).view(3, -1)
            Y_od[idx,:,:] = y_od
            
            img = ToNPArray(th.squeeze(ToUint8(model_kwargs['ref_img'][idx,:,:,:])))
            stain_matrix = th.from_numpy(get_stain_matrix(img).T).to(dist_util.dev())
            A[idx,:,:] = th.matmul(stain_matrix, stain_matrix_S_inv)
        AAInv = {'A' : A, 'Y_od' : Y_od}

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        import time
        start_time = time.time()
        sample = diffusion.ddim_sample_loop(
            model,
            (current_batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            trans_matrix=AAInv)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()
        
        for i in range(current_batch_size):
            out_path = os.path.join(logger.get_dir(), filename[i].split('/')[-1])
            img = Image.fromarray(np.squeeze(sample[i,:,:,:]))
            img.save(out_path)
        
        count += 1
        logger.log(f"Created {count * args.batch_size * dist_util.get_world_size()} samples")
        
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        base_samples='',
        save_dir='',
        stain_matrix_dir='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()