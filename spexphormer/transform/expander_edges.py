import math
import numpy as np
import scipy as sp
from pathlib import Path
from typing import Any, Optional
import torch
from spexphormer.transform.dist_transforms import laplacian_eigenv
from torch_geometric.graphgym.config import cfg


def generate_random_regular_graph2(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = rng.permutation(senders).tolist()

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  return senders, receivers


def generate_random_regular_graph1(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = []
  for _ in range(degree):
    receivers.extend(rng.permutation(list(range(num_nodes))).tolist())

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers


def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):
  """Generates a 2d-regular graph with n nodes using d random hamiltonian cycles.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = []
  receivers = []
  for _ in range(degree):
    permutation = rng.permutation(list(range(num_nodes))).tolist()
    for idx, v in enumerate(permutation):
      u = permutation[idx - 1]
      senders.extend([v, u])
      receivers.extend([u, v])

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers


def augment_with_expander(data, degree, algorithm, rng=None, max_num_iters=100, exp_index=0):
  """Generates a random d-regular expander graph with n nodes.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
    max_num_iters: maximum number of iterations
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  num_nodes = data.num_nodes

  if rng is None:
    rng = np.random.default_rng()
  
  eig_val = -1
  eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)

  max_eig_val_so_far = -1
  max_senders = []
  max_receivers = []
  cur_iter = 1

  if num_nodes <= degree:
    degree = num_nodes - 1    
    
  # if there are too few nodes, random graph generation will fail. in this case, we will
  # add the whole graph.
  if num_nodes <= 10:
    for i in range(num_nodes):
      for j in range(num_nodes):      
        if i != j:
          max_senders.append(i)
          max_receivers.append(j)
  else:
    while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
      if algorithm == 'Random-d':
        senders, receivers = generate_random_regular_graph1(num_nodes, degree, rng)
      elif algorithm == 'Random-d-2':
        senders, receivers = generate_random_regular_graph2(num_nodes, degree, rng)
      elif algorithm == 'Hamiltonian':
        senders, receivers = generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng)
      else:
        raise ValueError('prep.exp_algorithm should be one of the Random-d or Hamiltonian')
      
      if num_nodes > 1e5:
        max_senders = senders
        max_receivers = receivers
        break

      [eig_val, _] = laplacian_eigenv(senders, receivers, k=1, n=num_nodes)
      if len(eig_val) == 0:
        print("num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d" %(num_nodes, degree, cur_iter, max_num_iters, len(senders), len(receivers)))
        eig_val = 0
      else:
        eig_val = eig_val[0]

      if eig_val > max_eig_val_so_far:
        max_eig_val_so_far = eig_val
        max_senders = senders
        max_receivers = receivers

      cur_iter += 1

  # eliminate self loops.
  non_loops = [
      *filter(lambda i: max_senders[i] != max_receivers[i], range(0, len(max_senders)))
  ]

  senders = np.array(max_senders)[non_loops]
  receivers = np.array(max_receivers)[non_loops]

  max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
  max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)
  expander_edges = torch.cat([max_senders, max_receivers], dim=1)


  data.edge_index = torch.cat([data.edge_index, expander_edges.t()], dim=1)
  num_exp_edges = expander_edges.shape[0]

  edge_type = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
  edge_type[-num_exp_edges:] = 1

  # Adding self loops for complete Exphormer edges
  num_nodes = data.num_nodes
  self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
  data.edge_index = torch.cat((data.edge_index, self_loops), dim=1)
  self_loop_feats = torch.full((num_nodes,), 2, dtype=torch.long)
  edge_type = torch.cat((edge_type, self_loop_feats))

  edge_type = edge_type.contiguous()

  if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    num_new_edges = edge_type.shape[0] - data.edge_attr.shape[0]
    new_edge_attr = torch.zeros((num_new_edges, data.edge_attr.shape[1]), dtype=data.edge_attr.dtype)
    data.edge_attr = torch.cat((data.edge_attr, new_edge_attr), dim=0)
    edge_type_attr = edge_type
    data.edge_attr = torch.cat((edge_type_attr, data.edge_attr), dim=1)
  else:
    data.edge_attr = edge_type

  if cfg.prep.save_edges:
    edge_set_path = Path(f'EdgeSets/{cfg.dataset.name}')
    edge_set_path.mkdir(parents=True, exist_ok=True)
    np.save(edge_set_path / 'edges_exp.npy', data.edge_index.cpu().detach().numpy())
    np.save(edge_set_path / 'edge_attr_exp.npy', data.edge_attr.cpu().detach().numpy())
  else:
    if exp_index == 0:
      data.expander_edges = expander_edges
    else:
      attrname = f"expander_edges{exp_index}"
      setattr(data, attrname, expander_edges)
    
      
  return data

def load_edges(data):
  edge_set_path = Path(f'EdgeSets/{cfg.dataset.name}')
  
  if cfg.prep.num_edge_sets == 1:
    edges = np.load(edge_set_path / f'edges_{cfg.prep.edge_set_name}.npy')
    edge_attr = np.load(edge_set_path / f'edge_attr_{cfg.prep.edge_set_name}.npy')
    data.edge_index = torch.from_numpy(edges)
    data.edge_attr = torch.from_numpy(edge_attr)
  else:
    for idx in range(cfg.prep.num_edge_sets):
      edges = np.load(edge_set_path / f'edges_{cfg.prep.edge_set_name}_{idx}.npy')
      edge_attr = np.load(edge_set_path / f'edge_attr_{cfg.prep.edge_set_name}_{idx}.npy')
      setattr(data, f'edge_index_{idx}', torch.from_numpy(edges))
      setattr(data, f'edge_attr_{idx}', torch.from_numpy(edge_attr).long())
  
  return data