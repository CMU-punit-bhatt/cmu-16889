import torch

from pytorch3d.ops.knn import knn_points
from torch._C import device

# define losses
def voxel_loss(voxel_src,voxel_tgt):
  prob_loss = torch.nn.BCEWithLogitsLoss()(voxel_src, voxel_tgt)
	# implement some loss for binary voxel grids
  return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):

  # src_len = point_cloud_src.size(0)
  # tgt_len = point_cloud_tgt.size(0)

  src_tgt_nn = knn_points(point_cloud_src, point_cloud_tgt)
  tgt_src_nn = knn_points(point_cloud_tgt, point_cloud_src)

  loss_chamfer = torch.sum(src_tgt_nn.dists) + torch.sum(tgt_src_nn.dists)
	# implement chamfer loss from scratch
  return loss_chamfer

def smoothness_loss(mesh_src):

  V = mesh_src.verts_packed()
  L = mesh_src.laplacian_packed()
  
  # faces = mesh_src.faces_packed()
  # L = torch.zeros((V.size(0), V.size(0)))

  # # for face in faces:
  # L[faces[:, 0], faces[:, 0]] += 1
  # L[faces[:, 0], faces[:, 1]] -= 1
  # L[faces[:, 0], faces[:, 2]] -= 1
  # L[faces[:, 1], faces[:, 0]] -= 1
  # L[faces[:, 1], faces[:, 1]] += 1
  # L[faces[:, 1], faces[:, 2]] -= 1
  # L[faces[:, 2], faces[:, 0]] -= 1
  # L[faces[:, 2], faces[:, 1]] -= 1
  # L[faces[:, 2], faces[:, 2]] += 1

  loss_laplacian = torch.square(torch.linalg.norm(torch.matmul(L, V)))
	# implement laplacian smoothening loss
  return loss_laplacian