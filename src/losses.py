from mindspore import nn


class SegmentationLossCell(nn.Cell):
    def __init__(self, network, use_edge=False, edge_loss_weight=0.3, ignore_index=255):
        super().__init__(auto_prefix=False)
        self.network = network
        self.use_edge = bool(use_edge)
        self.edge_loss_weight = float(edge_loss_weight)
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.edge_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def construct(self, image, mask, edge):
        if self.use_edge:
            seg_logits, edge_logits = self.network(image)
            return self.seg_loss(seg_logits, mask) + self.edge_loss_weight * self.edge_loss(edge_logits, edge)
        logits = self.network(image)
        return self.seg_loss(logits, mask)
