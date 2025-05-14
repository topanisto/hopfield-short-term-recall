import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class HopfieldShortTerm(Hopfield):
  def __init__(self, num_nodes, max_iter):
    super().__init__(num_nodes, max_iter)

  def short_term_recall(self, inputs, c=1):
    seen_list = []
    for idx, input in enumerate(inputs):
      if idx == 0:
        self.fill_prototypes([input])
        seen_list.append(0)
        continue
      else:
        result, _ = self.recall(input)
        l = len(self.prototypes)
        h = hamming(result, input) #0 if hamming is 0 or 25
        p_seen = np.exp(- c*h * max(accuracy[l-1], 0.01)) #flip with this probability
        seen = np.random.choice([0, 1], p=[1-p_seen, p_seen])
        seen_list.append(seen)
        if not seen:
          self.prototypes = np.append(self.prototypes, [input], axis=0)
          # update weights and zero out diagonal
          self.weights += np.outer(input, input) / self.num_nodes
          np.fill_diagonal(self.weights, 0)
    return seen_list

class Hopfield:
  def __init__(self, num_nodes, max_iter):
    self.num_nodes = num_nodes
    self.weights = np.zeros((num_nodes, num_nodes))
    self.max_iter = max_iter

  def short_term_recall(self, inputs, c=1):
    return
  
  def fill_prototypes(self, prototypes):
    """
    Train weights on prototypes.
    The weights of a prototype are set to 
    """
    self.prototypes = prototypes
    n = len(prototypes)

    # hebbian correlation
    for i in range(n):
      t = prototypes[i]
      self.weights += np.outer(t, t)
    
    # zero out diagonal
    np.fill_diagonal(self.weights, 0)
    self.weights /= self.num_nodes


  def recall(self, state, threshold=None):
    """
    Predict prototype that state is closest to.

    threshold: a np.array of size self.num_nodes
    """
    if not threshold:
      self.threshold = np.zeros(self.num_nodes)
    else:
      self.threshold = threshold
    e = self.energy(state)
    fin = state

    for i in range(self.max_iter):
      fin, flip = self._async_update(fin)
      # if no state transitions, then exit loop
      if not flip:
        return fin, i
    
    # no convergence
    return fin, self.max_iter

  # Hopfield energy. assume no thresholds!
  def energy(self, state):
    energy = -0.5 * np.dot(state, np.dot(self.weights, state)) + np.dot(state, self.threshold)
  
  # asynchronously update a node
  def _async_update(self, state):
    """
    Update all nodes asynchronously until convergence.
    """
    # Xi = sign(sum(w_ijxj))
    s_new = state.copy()
    flip = False
    # randomly iterate through nodes
    order = np.random.permutation(self.num_nodes)
    for idx in order:
      new = np.sign(self.weights[idx].T @ s_new - self.threshold[idx])
      if new != s_new[idx]: # flip
        flip = True
      s_new[idx] = new
    return s_new, flip