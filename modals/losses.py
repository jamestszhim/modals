import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(d_real, d_fake, eps):
    return -torch.mean(torch.log(d_real+eps)+torch.log(1-d_fake+eps))


def adverserial_loss(d_fake, eps):
    return -torch.mean(torch.log(d_fake+eps))


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        # print(f'+ve: {ap_distances.mean()}\t-ve: {an_distances.mean()}')
        losses = F.relu(ap_distances - an_distances + self.margin)
        # losses = torch.max(an_distances - ap_distances+ self.margin, 0)[0]

        return losses.mean(), len(triplets)
