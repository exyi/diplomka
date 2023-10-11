import unittest, os, sys, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch, torch.nn as nn, torch.nn.functional as F
from . import torchutils, ntcnetwork, training
from model import hyperparameters

test_input = {
    "is_dna": torch.BoolTensor([ [ False, False, True, True ], [ True, True, True, False ] ]).to(torchutils.device),
    "sequence": torch.LongTensor([ [ 2, 1, 1, 1 ], [ 2, 2, 1, 0 ] ]).to(torchutils.device),
    "lengths": torch.LongTensor([ 4, 3 ]),
}

test_target = {
    "lengths": torch.LongTensor([ 4, 3 ]),
    "NtC": torch.LongTensor([ [ 6, 0, 3 ], [ 1, 1, 0 ] ]).to(torchutils.device),
}

class TestMyCode(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    

    def test_cross_entropy_usage(self):
        loss = torch.nn.CrossEntropyLoss(reduce=False)
        input = torch.tensor([0.1, 0.2, 0.7])
        target = torch.tensor(0)

        dim1 = loss(input, target)
        self.assertEqual(dim1.shape, ())
        input, target = input.unsqueeze(0).expand(5, 3), target.unsqueeze(0).expand(5)
        self.assertEqual(input.shape, (5, 3))
        self.assertEqual(target.shape, (5,))
        dim2 = loss(input, target)
        self.assertEqual(dim2.shape, (5,))
        self.assertAlmostEqual(dim1.item(), dim2[0])

        input, target = input.unsqueeze(-1).expand(5, 3, 2), target.unsqueeze(-1).expand(5, 2)
        self.assertEqual(input.shape, (5, 3, 2))
        self.assertEqual(target.shape, (5, 2))
        dim3 = loss(input, target)
        self.assertEqual(dim3.shape, (5, 2))
        self.assertAlmostEqual(dim2[0], dim3[0, 0])

    def test_metrics_mask(self):
        values = torch.tensor([ [ 1, 2, 3 ], [ 4, 5, 6 ] ])
        values_oh = F.one_hot(values, num_classes=10).transpose(1, 2)
        self.assertEqual(values_oh.shape, (2, 10, 3))
        lengths = torch.LongTensor([ 2, 3 ])

        mask = training.lengths_mask(lengths, values.shape[1])
        m_values = training.maskout_for_metrics(values, mask)
        m_values_oh = training.maskout_for_metrics(values_oh, mask, broadcast_dims=[False, True, False])
        self.assertEqual(m_values.shape, (5,))
        self.assertEqual(m_values_oh.shape, (5, 10))


    def test_model1(self):
        ntcnet = ntcnetwork.Network(hyperparameters.Hyperparams()).to(torchutils.device)
        out = ntcnet(test_input)
        self.assertEqual(out["NtC"].shape, (2, 97, 3))

        ntcs_y_pred, ntcs_y = training._output_field_transform("NtC", len_offset=1)({
            "y": test_target,
            "y_pred": out,
        })
        self.assertEqual(ntcs_y.shape, (5,))
        self.assertEqual(ntcs_y_pred.shape, (5, 97))

        ntcs_y_pred, ntcs_y = training._output_field_transform("NtC", len_offset=1, filter=training._NANT_filter)({
            "y": test_target,
            "y_pred": out,
        })
        self.assertTrue(ntcs_y.cpu().equal(torch.tensor([6, 3, 1, 1]).cpu()), f"{ntcs_y=}")
        self.assertEqual(ntcs_y_pred.shape, (4, 97))

if __name__ == '__main__':
    unittest.main()
