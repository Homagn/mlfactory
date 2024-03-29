import sys,os

# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re
try: #testing the functions locally without pip install
  import __init__
  cimportpath = os.path.abspath(__init__.__file__)
  if 'extensions' in cimportpath:
    print("local testing ")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)

except: #testing while mlfactory is installed using pip
  print("Non local testing")
  import mlfactory
  cimportpath = os.path.abspath(mlfactory.__file__)

main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("got main package location ",main_package_loc)


os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['top']))
#==========================================================




import torch, typing
import numpy as np

#from disk_features.disk import NpArray, Features, MatchedPairs
#from disk_features.disk.geom import distance_matrix

from applications.multiview_sba_recon.disk_features.disk import NpArray, Features, MatchedPairs
from applications.multiview_sba_recon.disk_features.disk.geom import distance_matrix


class CycleMatcher:
    def match_features(self, feat_1, feat_2):
        dist_m = distance_matrix(feat_1, feat_2)

        if dist_m.shape[0] == 0 or dist_m.shape[1] == 0:
            msg = '''
            Feature matching failed because one image has 0 detected features.
            This likely means that the algorithm has converged to a local
            optimum of detecting no features at all (0 reward). This can arise
            when lambda_fp and lambda_kp penalties are too high. Please check
            that your penalty annealing scheme is sound. It can also be that
            you are using a too low value of --warmup or --chunk-size
            '''
            raise RuntimeError(msg)

        n_amin = torch.argmin(dist_m, dim=1)
        m_amin = torch.argmin(dist_m, dim=0)

        # nearest neighbor's nearest neighbor
        nnnn = m_amin[n_amin]

        # we have a cycle consistent match for each `i` such that
        # nnnn[i] == i. We create an auxiliary array to check for that
        n_ix = torch.arange(dist_m.shape[0], device=dist_m.device)
        mask = nnnn == n_ix

        # Now `mask` is a binary mask and n_amin[mask] is an index array.
        # We use nonzero to turn `n_amin[mask]` into an index array and return
        return torch.stack([
            torch.nonzero(mask, as_tuple=False)[:, 0],
            n_amin[mask],
        ], dim=0)

    def match_pairwise(
        self,
        features: NpArray[Features], # [N_scenes, N_per_scene]
    ): # -> [N_scenes, (N_per_scene choose 2)]
        N_scenes, N_per_scene = features.shape
        N_combinations        = N_per_scene * (N_per_scene - 1) // 2

        matched_pairs = np.zeros((N_scenes, N_combinations), dtype=object)
        for i_scene, scene_f in enumerate(features):
            i_decision = 0

            for i in range(N_per_scene):
                features1 = scene_f[i]
                for j in range(i+1, N_per_scene):
                    features2 = scene_f[j]

                    matched_pairs[i_scene, i_decision] = MatchedPairs(
                        features1.kp,
                        features2.kp,
                        self.match_features(features1.desc, features2.desc),
                    )
                    
                    i_decision += 1

        return matched_pairs
