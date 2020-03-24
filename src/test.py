#!/usr/bin/env python
import os
import time
import cv2

import numpy as np
import torch

from config import *
from data_loader import EventData
from eval_utils import *
from vis_utils import *
from EVFlowNet import EVFlowNet

def drawImageTitle(img, title):
    cv2.putText(img,
                title,
                (60, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=2,
                bottomLeftOrigin=False)
    return img

def test(args, EVFlowNet_model, EventDataLoder):
    if args.test_plot:
        cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)

    if args.gt_path:
        print("Loading ground truth {}".format(args.gt_path))
        gt = np.load(args.gt_path)
        gt_timestamps = gt['timestamps']
        U_gt_all = gt['x_flow_dist']
        V_gt_all = gt['y_flow_dist']
        print("Ground truth loaded")
    
        AEE_sum = 0.
        percent_AEE_sum = 0.
        AEE_list = []

    if args.save_test_output:
        output_flow_list = []
        gt_flow_list = []
        event_image_list = []

    max_flow_sum = 0
    min_flow_sum = 0
    iters = 0

    for event_image, prev_image, next_image, image_timestamps in EventDataLoder:
        image_timestamps[0] = image_timestamps[0].numpy()
        image_timestamps[1] = image_timestamps[1].numpy()
        prev_image = prev_image.numpy()
        next_image = next_image.numpy()
        prev_image = np.transpose(prev_image, (0,2,3,1))
        next_image = np.transpose(next_image, (0,2,3,1))

        start_time = time.time()
        flow_dict = EVFlowNet_model(event_image.cuda())
        network_duration = time.time() - start_time
        
        pred_flow = np.squeeze(flow_dict['flow3'].detach().cpu().numpy())
        pred_flow = np.transpose(pred_flow, (1,2,0))
        pred_flow = np.flip(pred_flow, 2)

        max_flow_sum += np.max(pred_flow)
        min_flow_sum += np.min(pred_flow)
        
        event_count_image = torch.sum(event_image[:, :2, ...], dim=1).numpy()
        event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
        event_count_image = np.squeeze(event_count_image)

        if args.save_test_output:
            output_flow_list.append(pred_flow)
            event_image_list.append(event_count_image)
        
        if args.gt_path:
            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all,
                                                        gt_timestamps,
                                                        image_timestamps[0],
                                                        image_timestamps[1])
            
            gt_flow = np.stack((U_gt, V_gt), axis=2)

            if args.save_test_output:
                gt_flow_list.append(gt_flow)
            
            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2
            
            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]       
        
            # Calculate flow error.
            AEE, percent_AEE, n_points = flow_error_dense(gt_flow, 
                                                        pred_flow, 
                                                        event_count_image,
                                                        'outdoor' in args.test_sequence)
            AEE_list.append(AEE)
            AEE_sum += AEE
            percent_AEE_sum += percent_AEE
            
        iters += 1
        if iters % 100 == 0:
            print('-------------------------------------------------------')
            print('Iter: {}, time: {:f}, run time: {:.3f}s\n'
                'Mean max flow: {:.2f}, mean min flow: {:.2f}'
                .format(iters, image_timestamps[0][0], network_duration,
                        max_flow_sum / iters, min_flow_sum / iters))
            if args.gt_path:
                print('Mean AEE: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
                    .format(AEE_sum / iters,
                            percent_AEE_sum / iters,
                            n_points))

        # Prep outputs for nice visualization.
        if args.test_plot:
            pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            pred_flow_rgb = drawImageTitle(pred_flow_rgb, 'Predicted Flow')
            
            event_time_image = np.squeeze(np.amax(event_image[:, 2:, ...].numpy(), axis=1))
            event_time_image = (event_time_image * 255 / event_time_image.max()).astype(np.uint8)
            event_time_image = np.tile(event_time_image[..., np.newaxis], [1, 1, 3])
            
            event_count_image = np.tile(event_count_image[..., np.newaxis], [1, 1, 3])

            event_time_image = drawImageTitle(event_time_image, 'Timestamp Image')
            event_count_image = drawImageTitle(event_count_image, 'Count Image')
            
            prev_image = np.squeeze(prev_image)
            prev_image = prev_image * 255.
            prev_image = np.tile(prev_image[..., np.newaxis], [1, 1, 3])

            prev_image = drawImageTitle(prev_image, 'Grayscale Image')
            
            gt_flow_rgb = np.zeros(pred_flow_rgb.shape)
            errors = np.zeros(pred_flow_rgb.shape)

            gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow - No GT')
            errors = drawImageTitle(errors, 'Flow Error - No GT')
            
            if args.gt_path:
                errors = np.linalg.norm(gt_flow - pred_flow, axis=-1)
                errors[np.isinf(errors)] = 0
                errors[np.isnan(errors)] = 0
                errors = (errors * 255. / errors.max()).astype(np.uint8)
                errors = np.tile(errors[..., np.newaxis], [1, 1, 3])
                errors[event_count_image == 0] = 0

                if 'outdoor' in args.test_sequence:
                    errors[190:, :] = 0
                
                gt_flow_rgb = flow_viz_np(gt_flow[...,0], gt_flow[...,1])

                gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow')
                errors= drawImageTitle(errors, 'Flow Error')
                
            top_cat = np.concatenate([event_count_image, prev_image, pred_flow_rgb], axis=1)
            bottom_cat = np.concatenate([event_time_image, errors, gt_flow_rgb], axis=1)
            cat = np.concatenate([top_cat, bottom_cat], axis=0)
            cat = cat.astype(np.uint8)
            cv2.imshow('EV-FlowNet Results', cat)
            cv2.waitKey(1)
                
    print('Testing done. ')
    if args.gt_path:
        print('mean AEE {:02f}, mean %AEE {:02f}'
            .format(AEE_sum / iters, 
                    percent_AEE_sum / iters))
    if args.save_test_output:
        if args.gt_path:
            print('Saving data to {}_output_gt.npz'.format(args.test_sequence))
            np.savez('{}_output_gt.npz'.format(args.test_sequence),
                    output_flows=np.stack(output_flow_list, axis=0),
                    gt_flows=np.stack(gt_flow_list, axis=0),
                    event_images=np.stack(event_image_list, axis=0))
        else:
            print('Saving data to {}_output.npz'.format(args.test_sequence))
            np.savez('{}_output.npz'.format(args.test_sequence),
                    output_flows=np.stack(output_flow_list, axis=0),
                    event_images=np.stack(event_image_list, axis=0))


def main():        
    args = configs()
    args.load_path = os.path.join(args.load_path, args.training_instance)

    EVFlowNet_model = EVFlowNet(args).cuda()
    EVFlowNet_model.load_state_dict(torch.load(args.load_path+'/model91'))
    #para = np.load('D://p.npy').item()
    #EVFlowNet_model.load_state_dict(para)
    EventDataset = EventData(args.data_path, 'test', skip_frames=args.test_skip_frames)
    EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=1, shuffle=False)

    if not args.load_path:
        raise Exception("You need to set `load_path` and `training_instance`.")
    
    EVFlowNet_model.eval()
    '''
    event,pre,next_,_ = next(iter(EventDataLoader))
    flow = EVFlowNet_model(event.cuda())
    a = flow['flow3']
    x = a[0,0].detach().cpu().numpy()
    y = a[0,1].detach().cpu().numpy()
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = np.max(x[~np.isinf(x)])
    y[np.isnan(y)] = 0
    y[np.isinf(y)] = np.max(y[~np.isinf(y)])
    a = np.sqrt(x**2+y**2)
    b = np.arctan(y/x)
    b[np.isnan(b)] = 0
    b[np.isinf(b)] = np.max(b[~np.isinf(b)])
    a = 255*(a-np.min(a))/(np.max(a)-np.min(a))
    a = a.astype(np.uint8)
    b = 180*(b-np.min(b))/(np.max(b)-np.min(b))
    b = b.astype(np.uint8)
    c = 255*np.ones(a.shape).astype(np.uint8)
    a = np.stack((b,a,c),axis=2)
    a = cv2.cvtColor(a,cv2.COLOR_HSV2BGR)
    cv2.namedWindow('w')
    cv2.imshow('w',a)
    cv2.waitKey()
    '''
    test(args, EVFlowNet_model, EventDataLoader)


if __name__ == "__main__":
    main()
