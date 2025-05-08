# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse
from xml.dom import minidom

CAMERA_ID_TO_NAME = {
  1: "54138969",
  2: "55011271",
  3: "58860488",
  4: "60457274",
}

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c


def load_camera_params(w0, subject, camera):
  """Load h36m camera parameters

  Args
    w0: 300-long array read from XML metadata
    subect: int subject id
    camera: int camera id
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  # Get the 15 numbers for this subject and camera
  w1 = np.zeros(15)
  start = 6 * ((camera-1)*11 + (subject-1))
  w1[:6] = w0[start:start+6]
  w1[6:] = w0[(265+(camera-1)*9 - 1): (264+camera*9)]

  def rotationMatrix(r):    
    R1, R2, R3 = [np.zeros((3, 3)) for _ in range(3)]

    # [1 0 0; 0 cos(obj.Params(1)) -sin(obj.Params(1)); 0 sin(obj.Params(1)) cos(obj.Params(1))]
    R1[0:] = [1, 0, 0]
    R1[1:] = [0, np.cos(r[0]), -np.sin(r[0])]
    R1[2:] = [0, np.sin(r[0]),  np.cos(r[0])]

    # [cos(obj.Params(2)) 0 sin(obj.Params(2)); 0 1 0; -sin(obj.Params(2)) 0 cos(obj.Params(2))]
    R2[0:] = [ np.cos(r[1]), 0, np.sin(r[1])]
    R2[1:] = [0, 1, 0]
    R2[2:] = [-np.sin(r[1]), 0, np.cos(r[1])]

    # [cos(obj.Params(3)) -sin(obj.Params(3)) 0; sin(obj.Params(3)) cos(obj.Params(3)) 0; 0 0 1];%
    R3[0:] = [np.cos(r[2]), -np.sin(r[2]), 0]
    R3[1:] = [np.sin(r[2]),  np.cos(r[2]), 0]
    R3[2:] = [0, 0, 1]

    return (R1.dot(R2).dot(R3))
    
  R = rotationMatrix(w1)
  T = w1[3:6][:, np.newaxis]
  f = w1[6:8][:, np.newaxis]
  c = w1[8:10][:, np.newaxis]
  k = w1[10:13][:, np.newaxis]
  p = w1[13:15][:, np.newaxis]
  name = CAMERA_ID_TO_NAME[camera]

  return R, T, f, c, k, p, name

# def load_cameras( bpath='metadata.xml', subjects=[1,5,6,7,8,9,11] ):
#   """Loads the cameras of h36m

#   Args
#     bpath: path to hdf5 file with h36m camera data
#     subjects: List of ints representing the subject IDs for which cameras are requested
#   Returns
#     rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
#   """
#   print("Bpath: ", bpath)
#   rcams = {}
  
#   with h5py.File(bpath,'r') as hf:
#     for s in subjects:
#       for c in range(4): # There are 4 cameras in human3.6m
#         rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

#   return rcams

def load_cameras(bpath, subjects=[1,5,6,7,8,9,11]):
  """Loads the cameras of h36m

  Args
    bpath: path to xml file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  xmldoc = minidom.parse(bpath)
  string_of_numbers = xmldoc.getElementsByTagName('w0')[0].firstChild.data[1:-1]

  # Parse into floats
  w0 = np.array(list(map(float, string_of_numbers.split(" "))))

  assert len(w0) == 300

  for s in subjects:
    for c in range(4): # There are 4 cameras in human3.6m
      rcams[(s, c+1)] = load_camera_params(w0, s, c+1)

  return rcams


def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate

  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T

def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2