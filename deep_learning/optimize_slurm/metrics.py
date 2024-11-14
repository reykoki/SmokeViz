import torch

def compute_iou(pred, true, level, iou_dict):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5) * 1
    intersection = (pred + true == 2).sum()
    union = (pred + true >= 1).sum()
    iou = intersection / union
    iou_dict[level]['prev_int'] = intersection
    iou_dict[level]['prev_union'] = union
    if torch.isnan(iou) == False:
        iou_dict[level]['int'] += intersection
        iou_dict[level]['union'] += union
        print('{} density smoke gives: {} IoU'.format(level, iou))
        return iou_dict
    else:
        return iou_dict

def display_iou(iou_dict):
    high_iou = iou_dict['high']['int']/iou_dict['high']['union']
    med_iou = iou_dict['medium']['int']/iou_dict['medium']['union']
    low_iou = iou_dict['low']['int']/iou_dict['low']['union']
    iou = (iou_dict['high']['int'] + iou_dict['medium']['int'] + iou_dict['low']['int'])/(iou_dict['high']['union'] + iou_dict['medium']['union'] + iou_dict['low']['union'])
    print('OVERALL HIGH DENSITY SMOKE GIVES: {} IoU'.format(high_iou))
    print('OVERALL MEDIUM DENSITY SMOKE GIVES: {} IoU'.format(med_iou))
    print('OVERALL LOW DENSITY SMOKE GIVES: {} IoU'.format(low_iou))
    print('OVERALL OVER ALL DENSITY GIVES: {} IoU'.format(iou))

