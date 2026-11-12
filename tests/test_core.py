"""CPU checks for released weights, feature shapes, losses, and grouped data."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro/es_ngs"))
sys.path.insert(0, str(ROOT / "repro/es_ngs/qwen_exp/scripts"))
from model import ESNGSCritic, count_params, three_way_loss, listwise_ranking_loss, listwise_probability_loss
from train_student_kd import StateDataset, collate_pad
from jev_decision import JevDecisionHeads


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_deployment_forward(self):
        checkpoint = torch.load(ROOT / "checkpoints/es_ngs_dm.pt", map_location="cpu", weights_only=True)
        model = ESNGSCritic().eval()
        self.assertEqual(count_params(model), 205253)
        model.load_state_dict(checkpoint["model"], strict=True)
        row = StateDataset([1], ROOT / "data/training")[0]
        batch = {k: row[k] for k in ("X", "A", "action", "global", "mask")}
        with torch.inference_mode():
            a, b = model(batch), model(batch)
        for key in a:
            self.assertTrue(torch.isfinite(a[key]).all())
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_real_labels_and_backward(self):
        dataset = StateDataset([1, 121], ROOT / "data/training",
            soft_csv=ROOT / "data/training/exact_soft_labels.csv",
            teacher_csv=ROOT / "data/training/teacher_utilities_supplement.csv", teacher_name="jev_direct")
        rows = [dataset[i] for i in range(2)]
        batch = collate_pad(rows)
        self.assertEqual(batch["X"].shape[-2:], (32, 24))
        self.assertEqual(batch["A"].shape[-2:], (33, 33))
        self.assertTrue((batch["mask"][:, :, 0] == 1).all())
        model = ESNGSCritic().train()
        row = rows[0]
        pred = model({k: row[k] for k in ("X", "A", "action", "global", "mask")})
        valid = row["feasible"]
        value, _ = three_way_loss(pred, row["y"], row["y_drive"], row["y_swap"], row["y_sync"], valid)
        value += listwise_ranking_loss(pred["gain"], row["y"], valid)
        value += listwise_probability_loss(pred["gain"], row["soft"], torch.ones_like(valid))
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_qwen_heads_without_backbone(self):
        heads = JevDecisionHeads(16, 48)
        out = heads(torch.zeros(8,16), torch.zeros(8,48), torch.arange(8).float())
        self.assertEqual(set(out), {"choice","gain","drive","swap","sync","feasible"})
        self.assertTrue(all(x.shape == (8,) and torch.isfinite(x).all() for x in out.values()))


if __name__ == "__main__":
    unittest.main()
