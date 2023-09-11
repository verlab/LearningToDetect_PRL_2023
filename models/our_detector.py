import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
import numpy as np
from .unet_parts import *

class Our(nn.Module):
    def __init__(self, enc_channels = [1, 32, 64, 128], deformable_encoder = False, deformable_decoder = False, device=None):
        super().__init__()
        dec_channels = enc_channels[::-1]
        self.encoder = Encoder(enc_channels, deformable_encoder)
        self.decoder = Decoder(enc_channels, dec_channels, deformable_decoder)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
      
        return out
    
    def detect(self, img, max_kps, threshold_score=0.1, ): # numpy image

        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                img = img[..., 0]
            else:
                raise RuntimeError("Invalid image shape: " + str(img.shape))
        
        img2 = np.copy(img)
        img2 = img2[..., np.newaxis]
                
        imT = torch.Tensor(img).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            score_map = self(imT.to(self.device))
            keypoints, selection = self.score_to_keypoints(score_map.squeeze(0).squeeze(0), max_kps, threshold_score)

        return keypoints, score_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    
    def score_to_keypoints(self, score_map, max_kps=-1, threshold_score=0.1): # numpy image      
        with torch.no_grad():
            h, w = score_map.shape                

            # t0 = time.time()
            score_map = score_map.unsqueeze(0).unsqueeze(0)

            # nms
            # t1 = time.time()
            local_max = F.max_pool2d(score_map, 5, stride=1, padding=2)
            is_local_max = (score_map == local_max)
            del local_max

            # t2 = time.time()
            is_not_edge = self.edgeFilter(score_map)

            detected = torch.min(is_local_max, is_not_edge).squeeze(0).squeeze(0)
            score_map = score_map.squeeze(0).squeeze(0)

            points = score_map * detected
            filter = points > threshold_score
            
            selection = torch.zeros_like(points)
            selection[filter] = points[filter]

            # t3 = time.time()
            nonzeros = selection.nonzero()
            kps = nonzeros[..., [1, 0]].contiguous()
            scores = selection[nonzeros[..., 0], nonzeros[..., 1]]
            keypoints = torch.cat([kps, scores.unsqueeze(1)], dim=1).cpu().numpy()
            # t4 = time.time()

            idxs = sorted(range(len(keypoints)), key=lambda k: keypoints[k][2], reverse=True)
            
            keypoints = [keypoints[i] for i in idxs]
            
            if max_kps == -1:
                max_kps = len(keypoints)

            keypoints = np.asarray(keypoints[: min(max_kps, len(keypoints))])

            # t5 = time.time()

            # print(f"Time: {t1 - t0:.3f} {t2 - t1:.3f} {t3 - t2:.3f} {t4 - t3:.3f} {t5 - t4:.3f}")

            return keypoints, selection.cpu().detach().numpy()
    
    def edgeFilter(self, img, thresould=10):
        batch = img
        b, c, h, w = batch.size()

        dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3).to(img.device)
        dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3).to(img.device)
        djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3).to(img.device)


        dii = F.conv2d(
            batch.view(-1, 1, h, w), dii_filter.to(self.device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            img.view(-1, 1, h, w), dij_filter.to(self.device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), djj_filter.to(self.device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (thresould + 1) ** 2 / thresould
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        return is_not_edge
    