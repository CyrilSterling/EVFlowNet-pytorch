import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np

def warp_images_with_flow(images, flow):
    """
    Generates a prediction of an image given the optical flow, as in Spatial Transformer Networks.
    """
    dim3 = 0
    if images.dim() == 3:
        dim3 = 1
        images = images.unsqueeze(0)
        flow = flow.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]
    flow_x,flow_y = flow[:,0,...],flow[:,1,...]
    coord_x, coord_y = torch.meshgrid(torch.arange(height), torch.arange(width))

    pos_x = coord_x.reshape(height,width).type(torch.float32).cuda() + flow_x
    pos_y = coord_y.reshape(height,width).type(torch.float32).cuda() + flow_y
    pos_x = (pos_x-(height-1)/2)/((height-1)/2)
    pos_y = (pos_y-(width-1)/2)/((width-1)/2)

    pos = torch.stack((pos_y,pos_x),3).type(torch.float32)
    result = torch.nn.functional.grid_sample(images, pos, mode='bilinear', padding_mode='zeros')
    if dim3 == 1:
        result = result.squeeze()
        
    return result


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smoothness_loss(flow):
    """
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]
    
    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) +\
                      charbonnier_loss(flow_ucrop - flow_dcrop) +\
                      charbonnier_loss(flow_ulcrop - flow_drcrop) +\
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /= 4.
    
    return smoothness_loss


def compute_photometric_loss(prev_images, next_images, flow_dict):
    """
    Multi-scale photometric loss, as defined in equation (3) of the paper.
    """
    total_photometric_loss = 0.
    loss_weight_sum = 0.
    for i in range(len(flow_dict)):
        for image_num in range(prev_images.shape[0]):
            flow = flow_dict["flow{}".format(i)][image_num]
            height = flow.shape[1]
            width = flow.shape[2]
            
            prev_images_resize = F.to_tensor(F.resize(F.to_pil_image(prev_images[image_num].cpu()),
                                                    [height, width])).cuda()
            next_images_resize = F.to_tensor(F.resize(F.to_pil_image(next_images[image_num].cpu()),
                                                    [height, width])).cuda()
            
            next_images_warped = warp_images_with_flow(next_images_resize, flow)

            distance = next_images_warped - prev_images_resize
            photometric_loss = charbonnier_loss(distance)
            total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum

    return total_photometric_loss


class TotalLoss(torch.nn.Module):
    def __init__(self, smoothness_weight, weight_decay_weight=1e-4):
        super(TotalLoss, self).__init__()
        self._smoothness_weight = smoothness_weight
        self._weight_decay_weight = weight_decay_weight

    def forward(self, flow_dict, prev_image, next_image, EVFlowNet_model):
        # weight decay loss
        weight_decay_loss = 0
        for i in EVFlowNet_model.parameters():
            weight_decay_loss += torch.sum(i**2)/2*self._weight_decay_weight

        # smoothness loss
        smoothness_loss = 0
        for i in range(len(flow_dict)):
            smoothness_loss += compute_smoothness_loss(flow_dict["flow{}".format(i)])
        smoothness_loss *= self._smoothness_weight / 4.

        # Photometric loss.
        photometric_loss = compute_photometric_loss(prev_image,
                                                    next_image,
                                                    flow_dict)

        # Warped next image for debugging.
        #next_image_warped = warp_images_with_flow(next_image,
        #                                          flow_dict['flow3'])
        
        loss = weight_decay_loss + photometric_loss + smoothness_loss

        return loss

if __name__ == "__main__":
    '''
    a = torch.rand(7,7)
    b = torch.rand(7,7)
    flow = {}
    flow['flow0'] = torch.rand(1,2,3,3)
    loss = compute_photometric_loss(a,b,0,flow)
    print(loss)
    '''
    a = torch.rand(1,5,5).cuda()
    #b = torch.rand(5,5)*5
    b = torch.rand((5,5)).type(torch.float32).cuda()
    b.requires_grad = True
    #c = torch.rand(5,5)*5
    c = torch.rand((5,5)).type(torch.float32).cuda()
    c.requires_grad = True
    d = torch.stack((b,c),0)
    print(a)
    print(b)
    print(c)
    r = warp_images_with_flow(a,d)
    print(r)
    r = torch.mean(r)
    r.backward()
    print(b.grad)
    print(c.grad)