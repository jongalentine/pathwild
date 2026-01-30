"""
Checkpoint management for resumable data processing.

This module provides checkpoint functionality for long-running data processing jobs,
enabling resumption from the last successfully processed row after interruptions.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ProcessingCheckpoint:
    """
    Manages checkpoint state for data processing jobs.

    Checkpoints track:
    - Last successfully processed row index
    - Processing parameters (force, limit, workers, batch_size)
    - File checksum to detect input file changes
    - Error tracking for failed rows

    Usage:
        checkpoint_mgr = ProcessingCheckpoint(checkpoint_dir)
        checkpoint = checkpoint_mgr.load_or_create(dataset_path, total_rows, ...)

        for idx in range(checkpoint['start_row'], total_rows):
            # process row
            if idx % batch_size == 0:
                checkpoint_mgr.update(checkpoint, idx, processed_count, errors)

        checkpoint_mgr.complete(checkpoint)
    """

    VERSION = "1.0"

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, dataset_path: Path, test_mode: bool = False) -> Path:
        """
        Get checkpoint file path for a dataset.

        Args:
            dataset_path: Path to the dataset CSV file
            test_mode: If True, use separate checkpoint for test runs

        Returns:
            Path to checkpoint JSON file
        """
        # Extract dataset name from path (e.g., "combined_northern_bighorn_presence_absence.csv")
        stem = dataset_path.stem

        # Remove common prefixes/suffixes to get clean name
        name = stem.replace("combined_", "").replace("_presence_absence", "")

        if test_mode:
            return self.checkpoint_dir / f"integrate_features_{name}_test.json"
        return self.checkpoint_dir / f"integrate_features_{name}.json"

    def compute_file_checksum(self, file_path: Path, bytes_to_read: int = 4096) -> str:
        """
        Compute SHA256 checksum of first N bytes of a file.

        This is used to detect if the input file has changed since the checkpoint
        was created, which would invalidate the checkpoint.

        Args:
            file_path: Path to file
            bytes_to_read: Number of bytes to read for checksum (default 4KB)

        Returns:
            Checksum string in format "sha256:<hex>"
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read(bytes_to_read)
            checksum = hashlib.sha256(data).hexdigest()
            return f"sha256:{checksum}"
        except Exception as e:
            logger.warning(f"Failed to compute checksum for {file_path}: {e}")
            return ""

    def create(
        self,
        dataset_path: Path,
        total_rows: int,
        force: bool = False,
        limit: Optional[int] = None,
        workers: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Create a new checkpoint.

        Args:
            dataset_path: Path to the dataset CSV file
            total_rows: Total number of rows to process
            force: Whether force mode is enabled
            limit: Row limit if specified
            workers: Number of parallel workers
            batch_size: Batch size for progress saves

        Returns:
            New checkpoint dictionary
        """
        now = datetime.utcnow().isoformat() + 'Z'

        checkpoint = {
            "version": self.VERSION,
            "dataset_path": str(dataset_path.absolute()),
            "dataset_name": self._extract_dataset_name(dataset_path),
            "started_at": now,
            "updated_at": now,
            "total_rows": total_rows,
            "last_processed_row": -1,  # -1 means nothing processed yet
            "rows_processed": 0,
            "rows_skipped": 0,
            "rows_with_errors": 0,
            "force_mode": force,
            "limit": limit,
            "workers": workers,
            "batch_size": batch_size,
            "state": "in_progress",
            "error_indices": [],
            "checksum": self.compute_file_checksum(dataset_path)
        }

        # Save immediately
        checkpoint_path = self.get_checkpoint_path(dataset_path, limit is not None)
        self._save_to_file(checkpoint, checkpoint_path)

        logger.info(f"Created checkpoint: {checkpoint_path}")
        return checkpoint

    def load(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load existing checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint JSON file

        Returns:
            Checkpoint dictionary, or None if file doesn't exist or is invalid
        """
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            # Validate required fields
            required_fields = ['version', 'dataset_path', 'last_processed_row',
                             'total_rows', 'state']
            for field in required_fields:
                if field not in checkpoint:
                    logger.warning(f"Checkpoint missing required field: {field}")
                    return None

            # Store the path for later saves
            checkpoint['_checkpoint_path'] = str(checkpoint_path)

            return checkpoint

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted checkpoint file {checkpoint_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def load_or_create(
        self,
        dataset_path: Path,
        total_rows: int,
        force: bool = False,
        limit: Optional[int] = None,
        workers: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Load existing checkpoint or create new one.

        This is the main entry point for checkpoint management.

        Args:
            dataset_path: Path to the dataset CSV file
            total_rows: Total number of rows to process
            force: Whether force mode is enabled (deletes existing checkpoint)
            limit: Row limit if specified
            workers: Number of parallel workers
            batch_size: Batch size for progress saves

        Returns:
            Checkpoint dictionary (either loaded or newly created)
        """
        test_mode = limit is not None
        checkpoint_path = self.get_checkpoint_path(dataset_path, test_mode)

        # Force mode: delete existing checkpoint and start fresh
        if force:
            if checkpoint_path.exists():
                logger.info(f"Force mode: deleting existing checkpoint {checkpoint_path}")
                self.delete(checkpoint_path)
            return self.create(dataset_path, total_rows, force, limit, workers, batch_size)

        # Try to load existing checkpoint
        checkpoint = self.load(checkpoint_path)

        if checkpoint is None:
            # No valid checkpoint, create new one
            return self.create(dataset_path, total_rows, force, limit, workers, batch_size)

        # Validate checkpoint matches current run
        if not self.validate(checkpoint, dataset_path, total_rows, limit):
            logger.warning("Checkpoint validation failed, starting fresh")
            self.delete(checkpoint_path)
            return self.create(dataset_path, total_rows, force, limit, workers, batch_size)

        # Check if already completed
        if checkpoint['state'] == 'completed':
            logger.info(f"Checkpoint shows processing already completed")
            return checkpoint

        # Valid checkpoint for resumption
        start_row = checkpoint['last_processed_row'] + 1
        logger.info(
            f"Resuming from row {start_row:,} of {total_rows:,} "
            f"({start_row/total_rows*100:.1f}% already done)"
        )

        return checkpoint

    def save(self, checkpoint: Dict[str, Any]) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint dictionary to save
        """
        # Get checkpoint path from stored path or compute it
        if '_checkpoint_path' in checkpoint:
            checkpoint_path = Path(checkpoint['_checkpoint_path'])
        else:
            dataset_path = Path(checkpoint['dataset_path'])
            test_mode = checkpoint.get('limit') is not None
            checkpoint_path = self.get_checkpoint_path(dataset_path, test_mode)

        self._save_to_file(checkpoint, checkpoint_path)

    def _save_to_file(self, checkpoint: Dict[str, Any], checkpoint_path: Path) -> None:
        """
        Save checkpoint to a specific file path.

        Args:
            checkpoint: Checkpoint dictionary to save
            checkpoint_path: Path to save to
        """
        # Don't save internal fields
        save_data = {k: v for k, v in checkpoint.items() if not k.startswith('_')}

        try:
            # Write to temp file first, then rename (atomic on most filesystems)
            temp_path = checkpoint_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            temp_path.rename(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def update(
        self,
        checkpoint: Dict[str, Any],
        last_row: int,
        rows_processed: int,
        rows_skipped: int = 0,
        error_indices: Optional[List[int]] = None
    ) -> None:
        """
        Update checkpoint with current progress.

        Args:
            checkpoint: Checkpoint dictionary to update
            last_row: Index of last successfully processed row
            rows_processed: Total count of rows processed so far
            rows_skipped: Count of rows skipped (already had data)
            error_indices: List of row indices that failed
        """
        checkpoint['last_processed_row'] = last_row
        checkpoint['rows_processed'] = rows_processed
        checkpoint['rows_skipped'] = rows_skipped
        checkpoint['updated_at'] = datetime.utcnow().isoformat() + 'Z'

        if error_indices is not None:
            checkpoint['rows_with_errors'] = len(error_indices)
            # Keep only last 100 error indices to avoid huge checkpoint files
            checkpoint['error_indices'] = error_indices[-100:] if len(error_indices) > 100 else error_indices

        self.save(checkpoint)
        logger.debug(f"Checkpoint updated: row {last_row:,}, processed {rows_processed:,}")

    def complete(self, checkpoint: Dict[str, Any]) -> None:
        """
        Mark checkpoint as completed.

        Args:
            checkpoint: Checkpoint dictionary to complete
        """
        checkpoint['state'] = 'completed'
        checkpoint['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        self.save(checkpoint)

        logger.info(
            f"Checkpoint completed: {checkpoint['rows_processed']:,} processed, "
            f"{checkpoint.get('rows_skipped', 0):,} skipped, "
            f"{checkpoint.get('rows_with_errors', 0):,} errors"
        )

    def validate(
        self,
        checkpoint: Dict[str, Any],
        dataset_path: Path,
        total_rows: int,
        limit: Optional[int] = None
    ) -> bool:
        """
        Validate checkpoint matches current run parameters.

        Args:
            checkpoint: Checkpoint dictionary to validate
            dataset_path: Current dataset path
            total_rows: Current total rows
            limit: Current row limit

        Returns:
            True if checkpoint is valid for resumption
        """
        # Check dataset path matches
        if checkpoint.get('dataset_path') != str(dataset_path.absolute()):
            logger.warning("Checkpoint dataset path mismatch")
            return False

        # Check file hasn't changed (checksum) - but ONLY for fresh checkpoints.
        # If checkpoint is "in_progress" or "completed", the file was modified
        # in-place during processing, so the checksum will have changed. This is
        # expected, so skip validation for both states.
        if checkpoint.get('state') not in ('in_progress', 'completed'):
            current_checksum = self.compute_file_checksum(dataset_path)
            if checkpoint.get('checksum') and current_checksum:
                if checkpoint['checksum'] != current_checksum:
                    logger.warning("Input file has changed since checkpoint was created")
                    return False
        else:
            logger.debug(f"Skipping checksum validation for {checkpoint.get('state')} checkpoint (file modified during processing)")

        # Check total rows matches (could differ if file was modified)
        checkpoint_rows = checkpoint.get('limit') or checkpoint.get('total_rows')
        current_rows = limit or total_rows
        if checkpoint_rows != current_rows:
            logger.warning(
                f"Row count mismatch: checkpoint has {checkpoint_rows}, "
                f"current file has {current_rows}"
            )
            return False

        # Check last_processed_row is within bounds
        if checkpoint['last_processed_row'] >= current_rows:
            logger.warning("Checkpoint last_processed_row exceeds current row count")
            return False

        return True

    def delete(self, checkpoint_path: Path) -> None:
        """
        Delete checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file to delete
        """
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Deleted checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")

    def get_status(self, dataset_path: Path, test_mode: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint status without modifying it.

        Args:
            dataset_path: Path to dataset
            test_mode: Whether to check test checkpoint

        Returns:
            Status dictionary or None if no checkpoint exists
        """
        checkpoint_path = self.get_checkpoint_path(dataset_path, test_mode)
        checkpoint = self.load(checkpoint_path)

        if checkpoint is None:
            return None

        return {
            'state': checkpoint.get('state', 'unknown'),
            'last_processed_row': checkpoint.get('last_processed_row', -1),
            'total_rows': checkpoint.get('total_rows', 0),
            'rows_processed': checkpoint.get('rows_processed', 0),
            'rows_skipped': checkpoint.get('rows_skipped', 0),
            'rows_with_errors': checkpoint.get('rows_with_errors', 0),
            'started_at': checkpoint.get('started_at'),
            'updated_at': checkpoint.get('updated_at'),
            'progress_percent': (
                (checkpoint.get('last_processed_row', -1) + 1) /
                checkpoint.get('total_rows', 1) * 100
                if checkpoint.get('total_rows', 0) > 0 else 0
            )
        }

    def _extract_dataset_name(self, dataset_path: Path) -> str:
        """Extract clean dataset name from path."""
        stem = dataset_path.stem
        name = stem.replace("combined_", "").replace("_presence_absence", "")
        name = name.replace("_test", "")
        return name
