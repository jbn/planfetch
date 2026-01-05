"""Comprehensive tests for planfetch.lightning module."""

import sys
from unittest.mock import Mock, MagicMock, patch

import pytest


class TestS3PrefetchLightningCallback:
    """Test suite for S3PrefetchLightningCallback."""

    def test_import_without_lightning(self):
        """Test that module can be imported without pytorch_lightning."""
        # This test verifies the module handles missing Lightning gracefully
        # The module should not raise even if Lightning is not installed
        import planfetch.lightning  # noqa: F401

    def test_callback_exists_with_lightning(self):
        """Test callback class exists when Lightning is available."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback

            assert S3PrefetchLightningCallback is not None
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

    def test_callback_init_with_all_controllers(self):
        """Test callback initialization with all controllers."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        train_ctrl = Mock()
        val_ctrl = Mock()
        test_ctrl = Mock()

        callback = S3PrefetchLightningCallback(
            train_controller=train_ctrl,
            val_controller=val_ctrl,
            test_controller=test_ctrl,
        )

        assert callback.train_controller is train_ctrl
        assert callback.val_controller is val_ctrl
        assert callback.test_controller is test_ctrl

    def test_callback_init_with_partial_controllers(self):
        """Test callback initialization with only some controllers."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        train_ctrl = Mock()

        callback = S3PrefetchLightningCallback(train_controller=train_ctrl)

        assert callback.train_controller is train_ctrl
        assert callback.val_controller is None
        assert callback.test_controller is None

    def test_callback_init_with_no_controllers(self):
        """Test callback initialization with no controllers."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        callback = S3PrefetchLightningCallback()

        assert callback.train_controller is None
        assert callback.val_controller is None
        assert callback.test_controller is None

    def test_on_train_epoch_start_with_controller(self):
        """Test on_train_epoch_start calls controller.set_epoch."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        train_ctrl = Mock()
        callback = S3PrefetchLightningCallback(train_controller=train_ctrl)

        trainer = Mock()
        trainer.current_epoch = 5
        pl_module = Mock()

        callback.on_train_epoch_start(trainer, pl_module)

        train_ctrl.set_epoch.assert_called_once_with(5)

    def test_on_train_epoch_start_without_controller(self):
        """Test on_train_epoch_start does nothing without controller."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        callback = S3PrefetchLightningCallback()

        trainer = Mock()
        trainer.current_epoch = 5
        pl_module = Mock()

        # Should not raise
        callback.on_train_epoch_start(trainer, pl_module)

    def test_on_validation_epoch_start_with_controller(self):
        """Test on_validation_epoch_start calls controller.set_epoch."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        val_ctrl = Mock()
        callback = S3PrefetchLightningCallback(val_controller=val_ctrl)

        trainer = Mock()
        trainer.current_epoch = 3
        pl_module = Mock()

        callback.on_validation_epoch_start(trainer, pl_module)

        val_ctrl.set_epoch.assert_called_once_with(3)

    def test_on_validation_epoch_start_without_controller(self):
        """Test on_validation_epoch_start does nothing without controller."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        callback = S3PrefetchLightningCallback()

        trainer = Mock()
        trainer.current_epoch = 3
        pl_module = Mock()

        # Should not raise
        callback.on_validation_epoch_start(trainer, pl_module)

    def test_on_test_epoch_start_with_controller(self):
        """Test on_test_epoch_start calls controller.set_epoch."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        test_ctrl = Mock()
        callback = S3PrefetchLightningCallback(test_controller=test_ctrl)

        trainer = Mock()
        trainer.current_epoch = 0
        pl_module = Mock()

        callback.on_test_epoch_start(trainer, pl_module)

        test_ctrl.set_epoch.assert_called_once_with(0)

    def test_on_test_epoch_start_without_controller(self):
        """Test on_test_epoch_start does nothing without controller."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        callback = S3PrefetchLightningCallback()

        trainer = Mock()
        trainer.current_epoch = 0
        pl_module = Mock()

        # Should not raise
        callback.on_test_epoch_start(trainer, pl_module)

    def test_callback_inherits_from_pl_callback(self):
        """Test callback inherits from pl.Callback."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        callback = S3PrefetchLightningCallback()
        assert isinstance(callback, pl.Callback)

    def test_multiple_epochs(self):
        """Test callback works across multiple epochs."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        train_ctrl = Mock()
        callback = S3PrefetchLightningCallback(train_controller=train_ctrl)

        trainer = Mock()
        pl_module = Mock()

        for epoch in range(10):
            trainer.current_epoch = epoch
            callback.on_train_epoch_start(trainer, pl_module)

        assert train_ctrl.set_epoch.call_count == 10
        # Verify epochs were called in order
        calls = [call[0][0] for call in train_ctrl.set_epoch.call_args_list]
        assert calls == list(range(10))

    def test_different_controllers_for_different_phases(self):
        """Test different controllers are called for different training phases."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        train_ctrl = Mock()
        val_ctrl = Mock()
        test_ctrl = Mock()

        callback = S3PrefetchLightningCallback(
            train_controller=train_ctrl,
            val_controller=val_ctrl,
            test_controller=test_ctrl,
        )

        trainer = Mock()
        trainer.current_epoch = 2
        pl_module = Mock()

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_test_epoch_start(trainer, pl_module)

        train_ctrl.set_epoch.assert_called_once_with(2)
        val_ctrl.set_epoch.assert_called_once_with(2)
        test_ctrl.set_epoch.assert_called_once_with(2)

    def test_controller_is_optional_for_each_phase(self):
        """Test that each controller is optional and independent."""
        try:
            import pytorch_lightning as pl

            from planfetch.lightning import S3PrefetchLightningCallback
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        val_ctrl = Mock()
        callback = S3PrefetchLightningCallback(val_controller=val_ctrl)

        trainer = Mock()
        trainer.current_epoch = 1
        pl_module = Mock()

        # Train and test should not raise even without controllers
        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_test_epoch_start(trainer, pl_module)

        # Only val should have been called
        val_ctrl.set_epoch.assert_called_once_with(1)
