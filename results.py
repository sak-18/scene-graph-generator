import torch
import json
import numpy as np
"""
from IPython.display import Image, display

listOfImageNames = ['/content/drive/My Drive/Offnote/SceneGraph/Scene-Graph-Benchmark.pytorch/checkpoints/custom_images/dog.jpg','/content/drive/My Drive/Offnote/SceneGraph/Scene-Graph-Benchmark.pytorch/checkpoints/custom_images/giraffe.jpg']
for imageName in listOfImageNames:
    display(Image(filename=imageName))
"""
x=torch.load("checkpoints/upload_causal_motif_sgdet/custom_prediction.pytorch")

vg_sgg_dicts = json.load(open('datasets/vg/VG-SGG-dicts-with-attri.json'))
#print(vg_sgg_dicts.keys())

for id,ele in enumerate(x):#an image
    print('Image:{}'.format(id))
    """
    print('pred_labels:')
    print(len(ele.get_field('pred_labels')))#len 80
    print('pred_scores:')
    print(len(ele.get_field('pred_scores')))#len 80
    print('rel_pair_idxs')
    print(len(ele.get_field('rel_pair_idxs').tolist()))#len 6320
    print('pred_rel_scores:')
    print(ele.get_field('pred_rel_scores'))#logits for 51 predicates
    """
    #prediction relation triplet
    thres=0.2
    pred_rel_pair = ele.get_field('rel_pair_idxs').tolist()
    pred_rel_label = ele.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    mask = pred_rel_score > thres
    pred_rel_score = pred_rel_score[mask]
    pred_rel_label = pred_rel_label[mask]
    ma=np.array(mask)
    pr=np.array(pred_rel_pair)
    predpairs=pr[ma]
    pred_rels = [(i[0],j,i[1]) for i, j in zip(predpairs, pred_rel_label.tolist())] 
    for triplet in pred_rels:
        obj1label=vg_sgg_dicts["idx_to_label"][str(triplet[0])] if triplet[0]!=0 else "_background_"
        obj2label=vg_sgg_dicts["idx_to_label"][str(triplet[2])] if triplet[2]!=0 else "_background_"
        predlabel=vg_sgg_dicts["idx_to_predicate"][str(triplet[1])]        
        print(obj1label,"-",predlabel,"-",obj2label)
    print("*************************************")