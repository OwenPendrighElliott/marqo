import json
import logging
from abc import abstractmethod, ABC
from pathlib import Path
from marqo_test import MarqoTestCase


class BaseCompatibilityTestCase(MarqoTestCase, ABC):
    """
    Base class for backwards compatibility tests. Contains a prepare method that should be implemented by subclasses to
    add documents / prepare marqo state. Also contains methods to save and load results to/from a file so that
    test results can be compared across versions.
    """
    indexes_to_delete = []

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not hasattr(cls, 'logger'):
            cls.logger = logging.getLogger(cls.__name__)
            if not cls.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s')
                handler.setFormatter(formatter)
                cls.logger.addHandler(handler)
            cls.logger.setLevel(logging.INFO)

    @classmethod
    def get_results_file_path(cls):
        """Dynamically generate a unique file path based on the class name."""
        return Path(f"{cls.__qualname__}_stored_results.json")

    @classmethod
    def tearDownClass(cls) -> None:
        # A function that will be automatically called after each test call
        # This removes all the loaded models. It will also remove all the indexes inside a marqo instance.
        # Be sure to set the indexes_to_delete list with the indexes you want to delete, in the test class.
        cls.removeAllModels()
        if cls.indexes_to_delete:
            cls.delete_indexes(cls.indexes_to_delete)
            cls.logger.debug(f"Deleting indexes {cls.indexes_to_delete}")

    @classmethod
    def save_results_to_file(cls, results):
        """Save results to a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('w') as f:
            json.dump(results, f, indent=4)
        cls.logger.debug(f"Results saved to {filepath}")

    @classmethod
    def load_results_from_file(cls):
        """Load results from a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('r') as f:
            results = json.load(f)
        cls.logger.debug(f"Results loaded from {filepath}")
        return results

    @classmethod
    def delete_file(cls):
        """Delete the results file."""
        filepath = cls.get_results_file_path()
        filepath.unlink()
        cls.logger.debug(f"Results file deleted: {filepath}")

    @abstractmethod
    def prepare(self):
        """Prepare marqo state like adding documents"""
        pass

    @classmethod
    def set_logging_level(cls, level: str):
        log_level = getattr(logging, level.upper(), None)
        if log_level is None:
            raise ValueError(f"Invalid log level: {level}. Using current log level: {logging.getLevelName(cls.logger.level)}.")
        cls.logger.setLevel(log_level)
        for handler in cls.logger.handlers:
            handler.setLevel(log_level)
        cls.logger.info(f"Logging level changed to. {level.upper()}")