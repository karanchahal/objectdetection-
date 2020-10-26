import torch

def get_offsets(gt_boxes, ex_boxes):
    """
    This function returns the offsets that need to applied to gt_boxes to morph them into ex_boxes.
    Both boxes should be of the same shape, N x 4.
    The output would be offsets given in a torch tensor of size N x 4.
    """
    ex_width = ex_boxes[:, 2] - ex_boxes[:, 0]
    ex_height = ex_boxes[:, 3] - ex_boxes[:, 1]
    ex_center_x = ex_boxes[:, 0] + 0.5*ex_width
    ex_center_y = ex_boxes[:, 1] + 0.5*ex_height

    gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5*gt_width
    gt_center_y = gt_boxes[:, 1] + 0.5*gt_height


    delta_x = (gt_center_x - ex_center_x) / ex_width
    delta_y = (gt_center_y - ex_center_y) / ex_height
    delta_scaleX = torch.log(gt_width / ex_width)
    delta_scaleY = torch.log(gt_height / ex_height)

    offsets = torch.cat([delta_x.unsqueeze(0), 
                    delta_y.unsqueeze(0),
                    delta_scaleX.unsqueeze(0),
                    delta_scaleY.unsqueeze(0)],
                dim=0)
    return offsets.permute(1,0)
