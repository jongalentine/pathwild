"""
Tests for checkpoint management functionality.

These tests verify the ProcessingCheckpoint class works correctly for
resumable data processing.
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.checkpoint import ProcessingCheckpoint


class TestProcessingCheckpoint:
    """Tests for ProcessingCheckpoint class."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / '.checkpoints'
        self.checkpoint_mgr = ProcessingCheckpoint(self.checkpoint_dir)

        # Create a mock dataset file
        self.dataset_dir = Path(self.temp_dir) / 'processed'
        self.dataset_dir.mkdir(parents=True)
        self.dataset_path = self.dataset_dir / 'combined_test_dataset_presence_absence.csv'
        self.dataset_path.write_text('latitude,longitude,elevation\n1.0,2.0,100\n3.0,4.0,200\n')

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_dir_created(self):
        """Checkpoint directory is created on initialization."""
        assert self.checkpoint_dir.exists()

    def test_get_checkpoint_path(self):
        """Checkpoint path is derived correctly from dataset path."""
        path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        assert path.name == 'integrate_features_test_dataset.json'

        test_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=True)
        assert test_path.name == 'integrate_features_test_dataset_test.json'

    def test_compute_file_checksum(self):
        """File checksum is computed correctly."""
        checksum = self.checkpoint_mgr.compute_file_checksum(self.dataset_path)
        assert checksum.startswith('sha256:')
        assert len(checksum) > 10

        # Same file should produce same checksum
        checksum2 = self.checkpoint_mgr.compute_file_checksum(self.dataset_path)
        assert checksum == checksum2

    def test_create_checkpoint(self):
        """New checkpoint is created with correct fields."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000,
            force=False,
            limit=None,
            workers=4,
            batch_size=100
        )

        assert checkpoint['version'] == '1.0'
        assert checkpoint['total_rows'] == 1000
        assert checkpoint['last_processed_row'] == -1
        assert checkpoint['rows_processed'] == 0
        assert checkpoint['state'] == 'in_progress'
        assert checkpoint['force_mode'] is False
        assert checkpoint['workers'] == 4
        assert checkpoint['batch_size'] == 100
        assert 'checksum' in checkpoint
        assert 'started_at' in checkpoint

        # Checkpoint file should exist
        checkpoint_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        assert checkpoint_path.exists()

    def test_load_checkpoint(self):
        """Existing checkpoint is loaded correctly."""
        # Create a checkpoint
        self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        # Load it
        checkpoint_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        loaded = self.checkpoint_mgr.load(checkpoint_path)

        assert loaded is not None
        assert loaded['total_rows'] == 1000
        assert loaded['state'] == 'in_progress'

    def test_load_nonexistent_checkpoint(self):
        """Loading nonexistent checkpoint returns None."""
        checkpoint_path = self.checkpoint_dir / 'nonexistent.json'
        loaded = self.checkpoint_mgr.load(checkpoint_path)
        assert loaded is None

    def test_load_corrupted_checkpoint(self):
        """Loading corrupted checkpoint returns None."""
        checkpoint_path = self.checkpoint_dir / 'corrupted.json'
        checkpoint_path.write_text('not valid json')

        loaded = self.checkpoint_mgr.load(checkpoint_path)
        assert loaded is None

    def test_update_checkpoint(self):
        """Checkpoint is updated with progress."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        self.checkpoint_mgr.update(
            checkpoint,
            last_row=499,
            rows_processed=400,
            rows_skipped=100,
            error_indices=[10, 20, 30]
        )

        assert checkpoint['last_processed_row'] == 499
        assert checkpoint['rows_processed'] == 400
        assert checkpoint['rows_skipped'] == 100
        assert checkpoint['rows_with_errors'] == 3
        assert checkpoint['error_indices'] == [10, 20, 30]

        # Verify persisted to disk
        checkpoint_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        with open(checkpoint_path) as f:
            saved = json.load(f)
        assert saved['last_processed_row'] == 499

    def test_complete_checkpoint(self):
        """Checkpoint is marked as completed."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        self.checkpoint_mgr.update(checkpoint, last_row=999, rows_processed=1000)
        self.checkpoint_mgr.complete(checkpoint)

        assert checkpoint['state'] == 'completed'

        # Verify persisted
        checkpoint_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        with open(checkpoint_path) as f:
            saved = json.load(f)
        assert saved['state'] == 'completed'

    def test_validate_checkpoint_valid(self):
        """Valid checkpoint passes validation."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        is_valid = self.checkpoint_mgr.validate(
            checkpoint,
            dataset_path=self.dataset_path,
            total_rows=1000,
            limit=None
        )

        assert is_valid is True

    def test_validate_checkpoint_wrong_path(self):
        """Checkpoint with wrong dataset path fails validation."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        other_path = self.dataset_dir / 'other_dataset.csv'
        other_path.write_text('a,b\n1,2\n')

        is_valid = self.checkpoint_mgr.validate(
            checkpoint,
            dataset_path=other_path,
            total_rows=1000,
            limit=None
        )

        assert is_valid is False

    def test_validate_completed_checkpoint_skips_checksum(self):
        """Completed checkpoint skips checksum validation (file modified during processing)."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        self.checkpoint_mgr.complete(checkpoint)

        # Modify the file (simulates in-place feature updates during processing)
        self.dataset_path.write_text('completely,different,content\n1,2,3\n')

        is_valid = self.checkpoint_mgr.validate(
            checkpoint,
            dataset_path=self.dataset_path,
            total_rows=1000,
            limit=None
        )

        # Completed checkpoints should skip checksum validation since
        # processing modifies the file in-place
        assert is_valid is True

    def test_validate_in_progress_checkpoint_skips_checksum(self):
        """In-progress checkpoint skips checksum validation (file modified during processing)."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        # Checkpoint is in_progress, not completed

        # Modify the file (simulating in-place updates during processing)
        self.dataset_path.write_text('modified,during,processing\n1,2,3\n')

        # Should still be valid because in_progress checkpoints skip checksum validation
        is_valid = self.checkpoint_mgr.validate(
            checkpoint,
            dataset_path=self.dataset_path,
            total_rows=1000,
            limit=None
        )

        assert is_valid is True

    def test_validate_checkpoint_row_count_mismatch(self):
        """Checkpoint fails validation if row count differs."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        is_valid = self.checkpoint_mgr.validate(
            checkpoint,
            dataset_path=self.dataset_path,
            total_rows=2000,  # Different row count
            limit=None
        )

        assert is_valid is False

    def test_load_or_create_new(self):
        """load_or_create creates new checkpoint when none exists."""
        checkpoint = self.checkpoint_mgr.load_or_create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        assert checkpoint['state'] == 'in_progress'
        assert checkpoint['last_processed_row'] == -1

    def test_load_or_create_resume(self):
        """load_or_create resumes from existing checkpoint."""
        # Create and update a checkpoint
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        self.checkpoint_mgr.update(checkpoint, last_row=499, rows_processed=500)

        # Load or create should resume
        resumed = self.checkpoint_mgr.load_or_create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        assert resumed['last_processed_row'] == 499
        assert resumed['rows_processed'] == 500

    def test_load_or_create_force_deletes(self):
        """load_or_create with force=True deletes existing checkpoint."""
        # Create and update a checkpoint
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        self.checkpoint_mgr.update(checkpoint, last_row=499, rows_processed=500)

        # Force should start fresh
        forced = self.checkpoint_mgr.load_or_create(
            dataset_path=self.dataset_path,
            total_rows=1000,
            force=True
        )

        assert forced['last_processed_row'] == -1
        assert forced['rows_processed'] == 0

    def test_load_or_create_completed_skips(self):
        """load_or_create returns completed checkpoint without reprocessing."""
        # Create and complete a checkpoint
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        self.checkpoint_mgr.update(checkpoint, last_row=999, rows_processed=1000)
        self.checkpoint_mgr.complete(checkpoint)

        # Should return completed checkpoint
        loaded = self.checkpoint_mgr.load_or_create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        assert loaded['state'] == 'completed'

    def test_delete_checkpoint(self):
        """Checkpoint is deleted correctly."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        checkpoint_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        assert checkpoint_path.exists()

        self.checkpoint_mgr.delete(checkpoint_path)
        assert not checkpoint_path.exists()

    def test_get_status(self):
        """get_status returns correct information."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )
        self.checkpoint_mgr.update(checkpoint, last_row=499, rows_processed=400, rows_skipped=100)

        status = self.checkpoint_mgr.get_status(self.dataset_path)

        assert status['state'] == 'in_progress'
        assert status['last_processed_row'] == 499
        assert status['total_rows'] == 1000
        assert status['rows_processed'] == 400
        assert status['rows_skipped'] == 100
        assert status['progress_percent'] == 50.0

    def test_get_status_no_checkpoint(self):
        """get_status returns None when no checkpoint exists."""
        status = self.checkpoint_mgr.get_status(self.dataset_path)
        assert status is None

    def test_error_indices_truncated(self):
        """Error indices are truncated to last 100."""
        checkpoint = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000
        )

        # Create 150 error indices
        error_indices = list(range(150))

        self.checkpoint_mgr.update(
            checkpoint,
            last_row=500,
            rows_processed=350,
            error_indices=error_indices
        )

        assert len(checkpoint['error_indices']) == 100
        assert checkpoint['error_indices'][0] == 50  # Last 100 of 150
        assert checkpoint['rows_with_errors'] == 150

    def test_test_mode_separate_checkpoint(self):
        """Test mode uses separate checkpoint file."""
        # Create regular checkpoint
        regular = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=1000,
            limit=None
        )
        self.checkpoint_mgr.update(regular, last_row=999, rows_processed=1000)

        # Create test checkpoint
        test = self.checkpoint_mgr.create(
            dataset_path=self.dataset_path,
            total_rows=50,
            limit=50
        )
        self.checkpoint_mgr.update(test, last_row=25, rows_processed=26)

        # Verify separate files
        regular_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=False)
        test_path = self.checkpoint_mgr.get_checkpoint_path(self.dataset_path, test_mode=True)

        assert regular_path != test_path
        assert regular_path.exists()
        assert test_path.exists()

        # Verify different content
        with open(regular_path) as f:
            regular_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)

        assert regular_data['total_rows'] == 1000
        assert test_data['total_rows'] == 50


class TestCheckpointIntegration:
    """Integration tests for checkpoint with integrate_environmental_features."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_survives_restart(self):
        """Checkpoint survives process restart (simulated)."""
        checkpoint_dir = Path(self.temp_dir) / '.checkpoints'
        dataset_path = Path(self.temp_dir) / 'test.csv'
        dataset_path.write_text('a,b\n1,2\n3,4\n')

        # First "process" - create and update checkpoint
        mgr1 = ProcessingCheckpoint(checkpoint_dir)
        cp1 = mgr1.load_or_create(dataset_path, total_rows=1000)
        mgr1.update(cp1, last_row=500, rows_processed=501)

        # Simulate restart - new manager instance
        mgr2 = ProcessingCheckpoint(checkpoint_dir)
        cp2 = mgr2.load_or_create(dataset_path, total_rows=1000)

        # Should resume from where we left off
        assert cp2['last_processed_row'] == 500
        assert cp2['rows_processed'] == 501
