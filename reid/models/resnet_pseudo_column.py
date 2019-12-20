from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import numpy as np
import pdb
__all__ = ['resnet50_pseudo_column']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, T=1, dim = 256, num_parts=6):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN
        self.T = T
        self.reduce_dim = dim
        self.num_parts = num_parts
#        self.offset = ConvOffset2D(32)
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=False)
        param_dict = torch.load('/mnt/pretrained_model/resnet50-19c8e357.pth')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if ('classifier' in i) or ('fc' in i):
                continue
            self.base.state_dict()[i].copy_(param_dict[i])

#==========================add dilation=============================#
        if self.FCN:
            self.base.layer4[0].conv2.stride=(1,1)
            self.base.layer4[0].downsample[0].stride=(1,1)
# ================append conv for FCN==============================#
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout = dropout
            self.instance = nn.ModuleList()
            for i in range(self.num_parts+1):
                local_conv = nn.Linear(2048, self.num_features,bias=False)
                init.kaiming_normal_(local_conv.weight, mode='fan_out')

                local_bn = nn.BatchNorm1d(self.num_features)
                init.constant_(local_bn.weight,1)
                init.constant_(local_bn.bias,0)

                fc = nn.Linear(self.num_features, self.num_classes) 
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)


                self.instance.append(
                    nn.Sequential(
                        nn.Dropout(self.dropout),
                        local_conv,
                        local_bn,
                        #nn.ReLU(inplace=True),
                        #nn.Dropout(self.dropout),
                        fc
                        )
                    )

##---------------------------stripe1----------------------------------------------#

            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, self.num_parts, kernel_size=1, bias=True)
            init.kaiming_normal_(self.local_mask.weight, mode='fan_out')
            init.constant_(self.local_mask.bias, 0)



#===================================================================#

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
               # self.f_bn = nn.BatchNorm1d(2048)
               # init.constant_(self.f_bn.weight, 1)
               # init.constant_(self.f_bn.bias, 0)

                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
               # init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
               # self.classifier = nn.Linear(self.num_features, self.num_classes)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, inputs, rp, ratio, part_labels=None):
        x = inputs
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
#=======================FCN===============================#
        if self.FCN:
            x_global = F.avg_pool2d(x, (24,8))
#====================================================================================
            # part_labels = part_labels.unsqueeze(1).expand(x.size(0), self.num_parts, x.size(2))
            score = []
            for i in range(self.num_parts):
                score.append((part_labels == i).float())
            score = torch.cat([t.unsqueeze(1) for t in score], 1)
            pscore = score.sum(2)
            score = score / torch.clamp(pscore.unsqueeze(2).expand_as(score), 1e-12)
            score.requires_grad = False
            bb, cc, hh, ww = x.size()
            feat = x.unsqueeze(2).expand(bb, cc, self.num_parts, hh, ww) * score.unsqueeze(1).unsqueeze(4).expand(bb, cc, self.num_parts, hh, ww)
            feat = feat.sum(4).sum(3).unsqueeze(3)
            # feat = feat.mean(4).sum(3).unsqueeze(3)
            x = feat

            out0 = x / torch.clamp(x.norm(2, 1).unsqueeze(1).expand_as(x), min=1e-12)

            # x_cls = x_cls.mean(3, keepdim=False)
            # x_list = []
            # for i in range(self.num_parts):
                # x_list.append([])
            # for tensor, clip in zip(x_cls, rp):
                # part_feats = tensor.chunk(int(clip), 1)
                # for i in range(len(part_feats)):
                    # x_list[i].append(part_feats[i].mean(1, keepdim=False))
            # for i in range(self.num_parts):
                # x_list[i] = torch.stack(x_list[i])
            # x_list.append(x_global)
            # c = []
            # for tensor, branch in zip(x_list, self.instance):
                # tensor = tensor.contiguous().view(tensor.size(0), -1)
                # c.append(branch(tensor)) 

            x_list = list(x.chunk(x.size(2), 2))
            x_list.append(x_global)
            c = []
            for tensor, branch in zip(x_list, self.instance):
                tensor = tensor.contiguous().view(tensor.size(0), -1)
                c.append(branch(tensor))
            ps = score

            return out0, c, ps, pscore#, pool5#, orig_weight, pool5#, (, x1, x2, x3, x4, x5) #, glob#, c6, c7

#==========================================================#


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x
        out1 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
#        x = self.drop(x)
        if self.has_embedding:
            x = self.feat(x)
#            out2 = x
#            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.feat_bn(x)
            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)


        return out2, x
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)






def resnet50_pseudo_column(**kwargs):
    return ResNet(50, **kwargs)
