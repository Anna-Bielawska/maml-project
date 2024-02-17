import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def train_val_split(eps_data: List[torch.Tensor], k_shot: int, shuffle: bool = True) -> Dict[str, torch.Tensor]:
    """Split n-way data into k-shot support and query sets

    Args:
        eps_data: a list of 2 tensors limited to n_way classes:
            + first tensor: data
            + second tensor: labels
        k_shot: a number of training data per class
        shuffle: shuffle data before splitting

    Returns: a dictionary containing the data split
    """
    # get information of image size
    nc, iH, iW = eps_data[0][0].shape

    # get labels
    labels, num_classes = normalize_labels(labels=eps_data[1])
    v_shot = k_shot

    data = {
        'x_t': torch.empty(size=(num_classes, k_shot, nc, iH, iW), device=eps_data[0].device),
        'x_v': torch.empty(size=(num_classes, v_shot, nc, iH, iW), device=eps_data[0].device),
        'y_t': torch.empty(size=(num_classes * k_shot,), dtype=torch.int64, device=eps_data[1].device),
        'y_v': torch.empty(size=(num_classes * v_shot,), dtype=torch.int64, device=eps_data[1].device)
    }
    for cls_id in range(num_classes):
        X = eps_data[0][labels == cls_id]
        if shuffle:
            X = X[torch.randperm(X.shape[0])]  # randomly permute the images to get different ones each time

        data['x_t'][cls_id, :, :, :, :] = X[:k_shot]
        data['x_v'][cls_id, :, :, :, :] = X[k_shot : 2*k_shot]

        data['y_t'][k_shot * cls_id: k_shot * (cls_id + 1)] = torch.tensor(data=[cls_id] * k_shot, dtype=torch.int64, device=labels.device)
        data['y_v'][v_shot * cls_id: v_shot * (cls_id + 1)] = torch.tensor(data=[cls_id] * v_shot, dtype=torch.int64, device=labels.device)

    data['x_t'] = data['x_t'].view(num_classes * k_shot, nc, iH, iW)
    data['x_v'] = data['x_v'].view(num_classes * v_shot, nc, iH, iW)

    return data


def normalize_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Normalize a list of labels, for example:
    [11, 11, 20, 20, 60, 60, 6, 6] => [0, 0, 1, 1, 2, 2, 3, 3]
    """
    if labels.ndim > 1:
        raise ValueError("Input must be a 1d tensor, not {}".format(labels.ndim))

    out = torch.empty_like(input=labels, device=labels.device)

    label_dict = {}
    for i in range(labels.numel()):
        val = labels[i].item()

        if val not in label_dict:
            label_dict[val] = torch.tensor(data=len(label_dict), device=labels.device)

        out[i] = label_dict[val]

    return out, len(label_dict)


def get_indices(labels: torch.Tensor, class_names: List[int]) -> List[int]:
    """Get all indices (locations) of labels in class_names

    Args:
        labels: a tensor of dataset classes
        class_names: a list of desired classes (subset), it will define tasks

    Returns: a list containing indices of all desired classes.
    """
    indices =  []
    for i in range(labels.shape[0]):
        if labels[i] in class_names:
            indices.append(i)
    return indices


def get_random_task(data: Dict[str, torch.Tensor], n_way: int = 3, k_shot: int = 10) -> Tuple[torch.Tensor]:
    """Create a random n-way k-shot classification task

    Args:
        data: a dictionary of 2 tensors:
            + first tensor: X (data)
            + second tensor: labels
        n_way: a number of classes per task
        k_shot: a number of training data per class

    Returns: a tuple containing task data split into support and query sets
    """
    classes = list(np.unique(data['labels']))  # do not allow duplicates
    n_way = min(max(n_way, 2), len(classes))  # be sure to consider at least 2 classes and at most all of them
    k_shot = min(max(k_shot, 1), (data['labels'].numel() // len(classes)) // 2) # be sure to have at least 1 example per class and at most half of the possible examples
    random_classes = random.sample(classes, n_way)
    idx = get_indices(data['labels'], random_classes)
    task_i_x, task_i_y = data['X'][idx].unsqueeze(1), data['labels'][idx]
    subtask_data = train_val_split([task_i_x, task_i_y], k_shot=k_shot)

    perm_idx_support = torch.randperm(subtask_data['x_t'].shape[0])
    perm_idx_query = torch.randperm(subtask_data['x_v'].shape[0])

    # randomly shuffle each time for training purposes
    subtask_data['x_t'] = subtask_data['x_t'][perm_idx_support]  # Support set data
    subtask_data['y_t'] = subtask_data['y_t'][perm_idx_support]  # Support set labels normalized
    subtask_data['x_v'] = subtask_data['x_v'][perm_idx_query]  # Query set data
    subtask_data['y_v'] = subtask_data['y_v'][perm_idx_query]  # Query set labels normalized

    return subtask_data['x_t'], subtask_data['y_t'], subtask_data['x_v'], subtask_data['y_v']


def plot_task_examples(X_support, y_support, X_query, y_query):
    """Plot at most 12 examples of images with their noramlized labels from the Support and Query datasets
    (some classes may be omitted).

    Args:
        X_support: a tensor of images from the support set
        y_support: a tensor of labels from the support set
        X_query: a tensor of images from the query set
        y_query: a tensor of labels from the query set
    """
    titles = ['Support set examples', 'Query set examples']
    for idx, data in enumerate([(X_support, y_support), (X_query, y_query)]):
        num_of_images = data[0].shape[0]
        fig, ax = plt.subplots(ncols=min(num_of_images, 12), figsize=(20, 2.5))
        fig.suptitle(titles[idx], fontsize=16)

        for i in range(min(num_of_images, 12)):
          ax[i].imshow(data[0][i].permute(1,2,0), cmap='gray')
          ax[i].set_title(f"label={data[1][i]}")
          ax[i].axis('off')
        plt.show()


class MAML(nn.Module):
    def __init__(self, n_way: int, n_support: int, n_query: int, approx: bool=False):
        super(MAML, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1
        self.change_way = False

        self.loss_fn = None
        self.classifier = None

        self.n_task = 5  # number of subtasks to train the model
        self.task_update_num = 1  # one update per subtask T_i
        self.train_lr = 0.01

        self.approx = approx # first order approximation
        self.fast_params_list = []

    def forward(self, x: torch.Tensor):
        out  = self.classifier(x)
        return out

    def set_forward(self, X_support:torch.Tensor, y_support:torch.Tensor, X_query:torch.Tensor, y_query:torch.Tensor, is_feature: bool=False):
        assert is_feature == False, 'MAML do not support fixed feature'

        self.fast_params_list.clear()

        fast_parameters = list(self.parameters()) # the first gradient is based on original weight
        for weight in self.parameters():
            weight.fast = None

        self.zero_grad()
        accuracies = []
        for task_step in range(self.task_update_num):
            perm_idx_support = torch.randperm(X_support.shape[0])
            # randomly shuffle each time for training purposes
            X_support = X_support[perm_idx_support]  # Support set data
            y_support = y_support[perm_idx_support]  # Support set labels normalized

            scores = self.forward(X_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) 
            if self.approx:
                grad = [ g.detach()  for g in grad ] # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] # create an updated weight.fast
                fast_parameters.append(weight.fast) # gradients calculated here are based on the latest fast weights, but the graph will retain the link to old weight.fasts

            q_scores = self.forward(X_query)
            predictions  = q_scores.argmax(dim=1)
            positions = (predictions == y_query).float()
            accuracies.append((positions.sum() / predictions.size()[0]).item() * 100)

        scores = self.forward(X_query)
        return scores, accuracies

    def set_forward_loss(self, X_support:torch.Tensor, y_support:torch.Tensor, X_query:torch.Tensor, y_query:torch.Tensor):
        scores, accuracies = self.set_forward(X_support, y_support, X_query, y_query, is_feature = False)
        loss = self.loss_fn(scores, y_query)

        predictions  = scores.argmax(dim=1)
        positions = (predictions == y_query).float()
        task_accuracy = (positions.sum() / predictions.size()[0]) * 100

        return loss, accuracies + [task_accuracy], positions.squeeze()


if __name__ == "__main__":
    pass