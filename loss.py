import torch
import torch.nn.functional as F
import cv2
import numpy as np
# ------------ cats losses ----------


def bdrloss(prediction, label, radius, device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum(cost)


def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, std, ada, device='cuda'):
    # tracingLoss
    # tex_factor, bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # new_mask = mask * torch.exp(std * ada)
    # print('bce')
    # cost = torch.sum(torch.nn.functional.binary_cross_entropy(
    #     prediction.float(), label.float(), weight=new_mask.detach(), reduction='sum'))
    cost = torch.nn.functional.binary_cross_entropy(prediction.float(), label.float(),
                                                    weight=mask.detach(), reduction='none')
    cost = torch.sum(cost)
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + 4. * bdrcost + 0.01 * textcost


def rcf_loss(inputs, label):
    # 与bdcn_loss2的主要区别是有ignore像素
    # label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float()  # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
    inputs = torch.sigmoid(inputs)
    # gradient[mask == 1] = gradient[mask == 1] * (1.0 * num_negative / (num_positive + num_negative))
    # gradient[mask == 0] = gradient[mask == 0] * (1.1 * num_positive / (num_positive + num_negative))
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())
    # print("cost: ", cost)
    # print("gradient_cost:", gradient_cost)

    return cost


def RWCE(inputs, label, rtv_weight):
    # 与bdcn_loss2的主要区别是有ignore像素
    rtv_weight = 1-rtv_weight.float()/255
    mask = label.float()
    rtv_weight[mask != 0] = 0
    lamda = 3
    num_positive = lamda * torch.sum((label > 0.5).float()).float()  # ==1.
    num_negative = torch.sum((label == 0).float()).float() - 2*torch.sum((label > 0.5).float()).float()
    beta = num_negative / (num_positive + num_negative)
    k = 1
    alpha = lamda * num_positive / (num_positive + num_negative)
    mask[mask == 1] = beta
    mask[mask == 0] = alpha * (1 + k * rtv_weight[mask == 0])
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return cost


def rcf_mask3_loss(inputs, label, rtv_weight):
    # 与bdcn_loss2的主要区别是有ignore像素
    rtv_weight = rtv_weight.float()/255
    filt = torch.ones(1, 1, 3, 3)
    filt.requires_grad = False
    filt = filt.cuda()
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=1)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 2
    rtv_weight[mask != 0] = 0
    num_positive = torch.sum((mask > 0.5).float()).float()  # ==1.
    num_negative = torch.sum((mask == 0).float()).float()
    alpha = 1.0 * num_negative / (num_positive + num_negative)
    one_alpha = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 1] = one_alpha + one_alpha*rtv_weight[mask == 1]
    mask[mask == 0] = one_alpha + one_alpha*rtv_weight[mask == 0]
    mask[mask == 2] = alpha
    # mask[label == 2] = 0
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return cost
