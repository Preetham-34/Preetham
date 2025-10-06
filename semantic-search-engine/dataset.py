"""Dataset loading and management with batch support and extended fields."""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages documents dataset, supports large files and extended fields."""

    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.sample_data = []
        # Optionally load sample data or leave empty if large dataset is used

    def load_dataset(self, file_path: Optional[str] = None) -> Tuple[List[str], List[Dict]]:
        """
        Load documents from JSON file.
        Returns list of IDs and list of records (dicts).

        Args:
            file_path: Optional path to dataset JSON file

        Returns:
            Tuple (ids, records)
        """
        file_path = file_path or self.data_file_path
        data = []
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} documents from {file_path}")
            else:
                logger.warning(f"Dataset file {file_path} not found, returning empty dataset.")
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            data = []

        ids = []
        for i, doc in enumerate(data):
            if "id" in doc:
                ids.append(doc["id"])
            else:
                ids.append(f"doc_{i:06d}")

        return ids, data, title, metadata_list
dataset_manager = DatasetManager()
