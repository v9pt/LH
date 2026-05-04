"""
Unit Tests — ANFIS Core Engine
================================
Run from project root: python -m pytest tests/test_core.py -v
(conftest.py adds app/backend to sys.path automatically)
"""

import torch
import numpy as np
import pytest
import tempfile
import os

from core.anfis_core import ANFIS, ANFISTrainer, GaussianMF, normalize_strengths, compute_rule_strengths



class TestGaussianMF:
    def test_output_shape(self):
        mf = GaussianMF(n_inputs=4, n_rules=3)
        x  = torch.rand(8, 4)
        mu = mf(x)
        assert mu.shape == (8, 4, 3), f"Expected (8,4,3), got {mu.shape}"

    def test_output_range(self):
        """Gaussian MF outputs must be in (0, 1]."""
        mf = GaussianMF(n_inputs=3, n_rules=2)
        x  = torch.randn(100, 3)
        mu = mf(x)
        assert (mu >= 0).all() and (mu <= 1.0 + 1e-6).all(), \
            "MF outputs must be in [0, 1]"

    def test_peak_at_centre(self):
        """MF should output ≈1.0 when x equals the centre."""
        mf = GaussianMF(n_inputs=1, n_rules=1)
        # Set centre to 0.5, sigma to 0.3
        with torch.no_grad():
            mf.c[0, 0] = 0.5
            mf.sigma[0, 0] = 0.3
        x  = torch.tensor([[0.5]])
        mu = mf(x)
        assert abs(mu.item() - 1.0) < 1e-5, \
            f"Expected MF(centre) ≈ 1.0, got {mu.item()}"


class TestRuleStrengths:
    def test_rule_count(self):
        """Total rules = n_mfs^n_inputs."""
        for n_i, n_k in [(2, 2), (3, 2), (4, 3)]:
            mf = GaussianMF(n_inputs=n_i, n_rules=n_k)
            x  = torch.rand(5, n_i)
            mu = mf(x)
            w  = compute_rule_strengths(mu)
            expected = n_k ** n_i
            assert w.shape[1] == expected, \
                f"Expected {expected} rules, got {w.shape[1]}"

    def test_strengths_non_negative(self):
        mf = GaussianMF(n_inputs=2, n_rules=2)
        x  = torch.rand(10, 2)
        w  = compute_rule_strengths(mf(x))
        assert (w >= 0).all(), "Rule strengths must be non-negative"


class TestNormalization:
    def test_sums_to_one(self):
        w     = torch.rand(16, 9)    # 9 rules, batch=16
        w_bar = normalize_strengths(w)
        row_sums = w_bar.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), \
            "Normalized strengths must sum to 1 per sample"


class TestANFIS:
    def test_forward_shape(self):
        model = ANFIS(n_inputs=4, n_mfs=2)
        x     = torch.rand(8, 4)
        y     = model(x)
        assert y.shape == (8, 1), f"Expected (8,1), got {y.shape}"

    def test_rule_count_property(self):
        model = ANFIS(n_inputs=4, n_mfs=3)
        assert model.get_rule_count() == 81, \
            f"Expected 81 rules (3^4), got {model.get_rule_count()}"

    def test_output_deterministic(self):
        """Same input → same output (no stochastic layers)."""
        model = ANFIS(n_inputs=3, n_mfs=2)
        model.eval()
        x = torch.rand(5, 3)
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2), "ANFIS output must be deterministic"

    def test_membership_params_shape(self):
        model = ANFIS(n_inputs=4, n_mfs=3)
        params = model.get_membership_params()
        assert params['centers'].shape == (4, 3)
        assert params['sigmas'].shape  == (4, 3)


class TestANFISTrainer:
    def test_convergence_on_simple_function(self):
        """ANFIS should learn a simple linear function reasonably well."""
        torch.manual_seed(0)
        model   = ANFIS(n_inputs=2, n_mfs=2)
        trainer = ANFISTrainer(model, lr=5e-3)

        # Target: y = 0.5 * x1 + 0.3 * x2
        N  = 300
        x  = torch.rand(N, 2) * 2 - 1
        y  = (0.5 * x[:, 0:1] + 0.3 * x[:, 1:2]).clamp(-1, 1)

        history = trainer.fit(x, y, epochs=100, verbose=False)

        assert history[-1] < 0.05, \
            f"Final MSE {history[-1]:.4f} too high — training did not converge"

    def test_loss_decreases(self):
        """Training loss should generally decrease over epochs."""
        torch.manual_seed(42)
        model   = ANFIS(n_inputs=2, n_mfs=2)
        trainer = ANFISTrainer(model, lr=1e-3)
        x = torch.rand(200, 2)
        y = x.mean(dim=1, keepdim=True)
        history = trainer.fit(x, y, epochs=50, verbose=False)

        # Compare first 10 vs last 10 epochs
        early_avg = np.mean(history[:10])
        late_avg  = np.mean(history[-10:])
        assert late_avg <= early_avg, \
            f"Loss did not decrease: early={early_avg:.4f}, late={late_avg:.4f}"


class TestDarknessEstimator:
    """Integration test for darkness estimator."""

    def test_df_range(self):
        from core.darkness_estimator import DarknessEstimator
        rng = np.random.default_rng(1)
        N   = 300
        g   = rng.uniform(1.0, 5.0, N).astype(np.float32)
        t   = ((g - 1.0) / 4.0).reshape(-1, 1)
        X   = np.column_stack([
            np.clip(1 - g/5 + rng.normal(0, .05, N), 0, 1),
            np.clip(.3 - g/20, 0, 1),
            np.clip(g/5, 0, 1),
            np.clip(1 - g/8, 0, 1),
        ]).astype(np.float32)

        de = DarknessEstimator(n_mfs=2)
        de.train_from_arrays(X, t, epochs=50, verbose=False)

        # Test that estimate() always returns a value in [0, 1]
        # (sigmoid + clip applied inside estimate())
        test_cases = [
            np.array([[0.9, 0.3, 0.1, 0.9]], dtype=np.float32),   # bright
            np.array([[0.1, 0.05, 0.9, 0.3]], dtype=np.float32),  # dark
            np.array([[0.5, 0.2, 0.5, 0.6]], dtype=np.float32),   # mid
        ]
        for feats in test_cases:
            norm_feats = (feats - de._feature_mean) / de._feature_std
            import torch, math
            x_t = torch.from_numpy(norm_feats)
            de.model.eval()
            with torch.no_grad():
                raw = de.model(x_t).item()
            df = float(np.clip(1.0 / (1.0 + math.exp(-raw * 4.0)), 0.0, 1.0))
            assert 0.0 <= df <= 1.0, f"DF must be in [0,1], got {df}"

    def test_save_load(self):
        from core.darkness_estimator import DarknessEstimator
        rng = np.random.default_rng(2)
        N   = 200
        g   = rng.uniform(1.0, 5.0, N).astype(np.float32)
        t   = ((g - 1.0) / 4.0).reshape(-1, 1)
        X   = np.column_stack([np.clip(1-g/5, 0, 1), np.clip(.3-g/20, 0, 1),
                                np.clip(g/5, 0, 1), np.clip(1-g/8, 0, 1)]).astype(np.float32)


        de = DarknessEstimator(n_mfs=2)
        de.train_from_arrays(X, t, epochs=20, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'de.pt')
            de.save(path)

            de2 = DarknessEstimator(n_mfs=2)
            de2.load(path)

            assert de2._trained, "Loaded estimator should be marked as trained"
            assert np.allclose(de2._feature_mean, de._feature_mean), \
                "Feature mean must survive save/load"


class TestLCR:
    """Tests for ANFIS-LCR module."""

    def test_lcr_encode_shape(self):
        from core.anfis_lcr import lcr_encode
        d, K = 48, 10
        rng  = np.random.default_rng(0)
        D    = rng.random((d, K)).astype(np.float32)
        p    = rng.random(d).astype(np.float32)
        alpha = lcr_encode(p, D, lam=1e-4)
        assert alpha.shape == (K,), f"Expected ({K},), got {alpha.shape}"

    def test_hallucinate_upscale(self):
        from core.anfis_lcr import FaceDictionary, ANFISLocalityRepresentation
        rng = np.random.default_rng(0)
        P, K, C, scale = 4, 5, 3, 2
        d    = P * P * C
        fdict = FaceDictionary(n_atoms=K, patch_size=P, stride=2, scale=scale)
        fdict.D_LR = rng.random((d, K)).astype(np.float32)
        fdict.D_HR = rng.random((d * scale**2, K)).astype(np.float32)
        fdict.atom_darknesses = rng.uniform(0, 1, K).astype(np.float32)
        fdict._built = True

        h = ANFISLocalityRepresentation(fdict)
        lr_img = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        hr_out = h.hallucinate(lr_img, darkness_factor=0.5)

        assert hr_out.shape == (16, 16, 3), \
            f"Expected (16,16,3), got {hr_out.shape}"
        assert hr_out.min() >= 0 and hr_out.max() <= 1.0 + 1e-5


class TestRegression:
    """Tests for regression reconstructor."""

    def test_reconstruct_shape(self):
        from core.regression_reconstructor import PositionPatchRegressor
        rng = np.random.default_rng(0)
        reg = PositionPatchRegressor(
            image_size_lr=(8, 8), patch_size=2, stride=2, scale=4, kernel='linear')
        imgs = [(rng.random((32, 32, 3)) * 255).astype(np.uint8)
                for _ in range(15)]
        reg.train(imgs, verbose=False)
        lr = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        hr = reg.reconstruct(lr)
        assert hr.shape == (32, 32, 3), f"Expected (32,32,3), got {hr.shape}"

    def test_blend_output_range(self):
        from core.regression_reconstructor import blend_lcr_and_regression
        rng  = np.random.default_rng(0)
        a    = rng.random((32, 32, 3)).astype(np.float32)
        b    = rng.random((32, 32, 3)).astype(np.float32)
        out  = blend_lcr_and_regression(a, b, darkness_factor=0.6)
        assert out.min() >= 0 and out.max() <= 1.0 + 1e-5


class TestMotionBlur:
    """Tests for motion blur handler."""

    def test_kernel_construction(self):
        from core.motion_blur_handler import _make_motion_blur_kernel
        k = _make_motion_blur_kernel(15, angle_deg=45.0, length=7)
        assert k.shape == (15, 15)
        assert abs(k.sum() - 1.0) < 1e-5, \
            f"Kernel must sum to 1.0, got {k.sum()}"

    def test_wiener_shape(self):
        from core.motion_blur_handler import wiener_deconvolve, _make_motion_blur_kernel
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 3)).astype(np.float32)
        k   = _make_motion_blur_kernel(7, 30.0, 3)
        out = wiener_deconvolve(img, k, snr=0.01)
        assert out.shape == (32, 32, 3)

    def test_severity_bright_vs_dark(self):
        from core.motion_blur_handler import detect_blur_severity
        import numpy as np
        # Sharp image (random noise has high Laplacian variance)
        sharp = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Blurred (uniform image has near-zero Laplacian variance)
        blurred = np.ones((64, 64, 3), dtype=np.uint8) * 128
        s_sharp   = detect_blur_severity(sharp)
        s_blurred = detect_blur_severity(blurred)
        assert s_blurred > s_sharp, \
            f"Blurred image should have higher severity. Got sharp={s_sharp:.3f}, blurred={s_blurred:.3f}"
