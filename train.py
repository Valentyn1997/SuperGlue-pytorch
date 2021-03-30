
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import SparseDataset
import os
import torch.multiprocessing
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import torch
from omegaconf import DictConfig

from models.superpoint import SuperPoint
from models.superglue import SuperGlue, SuperGlueLightning
from models.matchingForTraining import MatchingForTraining
import logging

logging.basicConfig(level=logging.INFO)

torch.multiprocessing.set_sharing_strategy('file_system')

# parser = argparse.ArgumentParser(
#     description='Image pair matching and pose evaluation with SuperGlue',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument(
#     '--viz', action='store_true',
#     help='Visualize the matches and dump the plots')
# parser.add_argument(
#     '--eval', action='store_true',
#     help='Perform the evaluation'
#             ' (requires ground truth pose and intrinsics)')

# parser.add_argument(
#     '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
#     help='SuperGlue weights')
# parser.add_argument(
#     '--max_keypoints', type=int, default=1024,
#     help='Maximum number of keypoints detected by Superpoint'
#             ' (\'-1\' keeps all keypoints)')
# parser.add_argument(
#     '--keypoint_threshold', type=float, default=0.005,
#     help='SuperPoint keypoint detector confidence threshold')
# parser.add_argument(
#     '--nms_radius', type=int, default=4,
#     help='SuperPoint Non Maximum Suppression (NMS) radius'
#     ' (Must be positive)')
# parser.add_argument(
#     '--sinkhorn_iterations', type=int, default=20,
#     help='Number of Sinkhorn iterations performed by SuperGlue')
# parser.add_argument(
#     '--match_threshold', type=float, default=0.2,
#     help='SuperGlue match threshold')

# parser.add_argument(
#     '--resize', type=int, nargs='+', default=[640, 480],
#     help='Resize the input image before running inference. If two numbers, '
#             'resize to the exact dimensions, if one number, resize the max '
#             'dimension, if -1, do not resize')
# parser.add_argument(
#     '--resize_float', action='store_true',
#     help='Resize the image after casting uint8 to float')

# parser.add_argument(
#     '--cache', action='store_true',
#     help='Skip the pair if output .npz files are already found')
# parser.add_argument(
#     '--show_keypoints', action='store_true',
#     help='Plot the keypoints in addition to the matches')
# parser.add_argument(
#     '--fast_viz', action='store_true',
#     help='Use faster image visualization based on OpenCV instead of Matplotlib')
# parser.add_argument(
#     '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
#     help='Visualization file extension. Use pdf for highest-quality.')

# parser.add_argument(
#     '--opencv_display', action='store_true',
#     help='Visualize via OpenCV before saving output images')
# parser.add_argument(
#     '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
#     help='Path to the list of image pairs for evaluation')
# parser.add_argument(
#     '--shuffle', action='store_true',
#     help='Shuffle ordering of pairs before processing')
# parser.add_argument(
#     '--max_length', type=int, default=-1,
#     help='Maximum number of pairs to evaluate')

# parser.add_argument(
#     '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
#     help='Path to the directory that contains the images')
# parser.add_argument(
#     '--eval_output_dir', type=str, default='dump_match_pairs/',
#     help='Path to the directory in which the .npz results and optional,'
#             'visualizations are written')
# parser.add_argument(
#     '--learning_rate', type=int, default=0.0001,
#     help='Learning rate')

# parser.add_argument(
#     '--batch_size', type=int, default=1,
#     help='batch_size')
# parser.add_argument(
#     '--train_path', type=str, default='/home/yinxinjia/yingxin/dataset/COCO2014_train/',
#     help='Path to the directory of training imgs.')
# parser.add_argument(
#     '--epoch', type=int, default=20,
#     help='Number of epoches')


@hydra.main(config_path='configs', config_name='config.yaml')
def main(args: DictConfig):
    # torch.autograd.set_detect_anomaly(True)

    model = SuperGlueLightning(args)
    mlf_logger = MLFlowLogger(experiment_name='SuperGlue', tracking_uri=args.exp.mlflow_uri)
    mlf_logger.experiment.log_param(mlf_logger.run_id, 'chekpoints_path', f'{os.getcwd()}/{args.exp.checkpoint_path}')
    checkpoint_callback = ModelCheckpoint(dirpath=args.exp.checkpoint_path, verbose=True, save_weights_only=True)

    trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                      logger=mlf_logger if args.exp.logging else None,
                      max_epochs=args.exp.epochs,
                      accumulate_grad_batches=args.exp.accumulate_grad_batches,
                      checkpoint_callback=checkpoint_callback if args.exp.checkpoint else None,
                      val_check_interval=0.1,
                      # limit_val_batches=args.data.val_size,
                      accelerator='dp',
                      log_every_n_steps=100,
                      num_sanity_val_steps=0,
                      )

    trainer.fit(model)


if __name__ == '__main__':
    main()
    # if torch.cuda.is_available():
    #     superglue.cuda() # make sure it trains on GPU
    # else:
    #     print("### CUDA not available ###")

    # start training
    # for epoch in range(1, opt.epoch+1):
    #     epoch_loss = 0
    #     superglue.double().train()
    #     for i, pred in enumerate(train_loader):
    #         for k in pred:
    #             if k != 'file_name' and k!='image0' and k!='image1':
    #                 if type(pred[k]) == torch.Tensor:
    #                     pred[k] = Variable(pred[k].cuda())
    #                 else:
    #                     pred[k] = Variable(torch.stack(pred[k]).cuda())
                
            # data = superglue(pred)
            # for k, v in pred.items():
            #     pred[k] = v[0]
            # pred = {**pred, **data}
            #
            # if pred['skip_train'] == True: # image has no keypoint
            #     continue
            #
            # # process loss
            # Loss = pred['loss']
            # epoch_loss += Loss.item()
            # mean_loss.append(Loss)
            #
            # superglue.zero_grad()
            # Loss.backward()
            # optimizer.step()

            # for every 50 images, print progress and visualize the matches
            # if (i+1) % 50 == 0:
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #         .format(epoch, opt.epoch, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item()))
            #     mean_loss = []
            #
            #     ### eval ###
            #     # Visualize the matches.
            #     superglue.eval()
            #     image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
            #     kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
            #     matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
            #     image0 = read_image_modified(image0, opt.resize, opt.resize_float)
            #     image1 = read_image_modified(image1, opt.resize, opt.resize_float)
            #     valid = matches > -1
            #     mkpts0 = kpts0[valid]
            #     mkpts1 = kpts1[matches[valid]]
            #     mconf = conf[valid]
            #     viz_path = eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
            #     color = cm.jet(mconf)
            #     stem = pred['file_name']
            #     text = []
            #
            #     make_matching_plot(
            #         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            #         text, viz_path, stem, stem, opt.show_keypoints,
            #         opt.fast_viz, opt.opencv_display, 'Matches')

            # process checkpoint for every 5e3 images
        #     if (i+1) % 5e3 == 0:
        #         model_out_path = "model_epoch_{}.pth".format(epoch)
        #         torch.save(superglue, model_out_path)
        #         print ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}'
        #             .format(epoch, opt.epoch, i+1, len(train_loader), model_out_path))
        #
        # # save checkpoint when an epoch finishes
        # epoch_loss /= len(train_loader)
        # model_out_path = "model_epoch_{}.pth".format(epoch)
        # torch.save(superglue, model_out_path)
        # print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
        #     .format(epoch, opt.epoch, epoch_loss, model_out_path))
        

