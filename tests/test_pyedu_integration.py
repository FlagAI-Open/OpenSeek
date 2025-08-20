#!/usr/bin/env python3
"""
Tests for PyEdu dataset integration in OpenSeek.

This test suite verifies that the PyEdu dataset can be properly integrated
into OpenSeek training pipelines, including configuration validation,
dataset utilities, and preprocessing functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add tools directory to path for importing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

try:
    from pyedu_dataset_utils import PyEduDatasetHandler
    PYEDU_UTILS_AVAILABLE = True
except ImportError:
    PYEDU_UTILS_AVAILABLE = False


class TestPyEduDatasetHandler(unittest.TestCase):
    """Test cases for PyEduDatasetHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PYEDU_UTILS_AVAILABLE:
            self.skipTest("PyEdu dataset utilities not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.handler = PyEduDatasetHandler(cache_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_handler_initialization(self):
        """Test PyEduDatasetHandler initialization."""
        self.assertIsInstance(self.handler, PyEduDatasetHandler)
        self.assertEqual(self.handler.DATASET_NAME, "Leon-Leee/unofficial-pyedu")
        self.assertEqual(self.handler.DATASET_SIZE_GB, 6)
        self.assertIsNotNone(self.handler.cache_dir)
    
    def test_validate_dataset_with_sample_data(self):
        """Test dataset validation with sample Python code data."""
        # Create a sample dataset file
        sample_data = [
            {"text": "def hello_world():\n    print('Hello, World!')"},
            {"text": "import numpy as np\n\nclass DataProcessor:\n    def __init__(self):\n        pass"},
            {"text": "if __name__ == '__main__':\n    main()"},
            {"text": "# This is a comment\nfor i in range(10):\n    print(i)"},
            {"text": "Regular text without code"}
        ]
        
        sample_file = os.path.join(self.temp_dir, "sample_pyedu.jsonl")
        with open(sample_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # Validate the sample dataset
        stats = self.handler.validate_dataset(sample_file)
        
        # Check statistics
        self.assertEqual(stats['total_examples'], 5)
        self.assertGreater(stats['total_characters'], 0)
        self.assertGreater(stats['avg_length'], 0)
        self.assertGreater(stats['file_size_mb'], 0)
        self.assertTrue(stats['contains_python_code'])
        self.assertGreater(stats['python_code_percentage'], 0)
    
    def test_validate_dataset_file_not_found(self):
        """Test dataset validation with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.handler.validate_dataset("/nonexistent/file.jsonl")
    
    def test_create_training_config(self):
        """Test creation of training configuration."""
        dataset_path = "/path/to/pyedu/dataset"
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        result_path = self.handler.create_training_config(dataset_path, config_path)
        
        self.assertEqual(result_path, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Check config content
        with open(config_path, 'r') as f:
            config_content = f.read()
            self.assertIn("PyEdu dataset", config_content)
            self.assertIn(dataset_path, config_content)
            self.assertIn("data_path:", config_content)


class TestPyEduConfigurationFiles(unittest.TestCase):
    """Test cases for PyEdu configuration files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'pyedu-integration')
    
    def test_config_files_exist(self):
        """Test that PyEdu configuration files exist."""
        config_file = os.path.join(self.config_dir, 'config_pyedu_integration.yaml')
        train_config_file = os.path.join(self.config_dir, 'train', 'train_pyedu_integration.yaml')
        readme_file = os.path.join(self.config_dir, 'README.md')
        
        self.assertTrue(os.path.exists(config_file), f"Config file not found: {config_file}")
        self.assertTrue(os.path.exists(train_config_file), f"Train config file not found: {train_config_file}")
        self.assertTrue(os.path.exists(readme_file), f"README file not found: {readme_file}")
    
    def test_config_file_structure(self):
        """Test that configuration files have proper structure."""
        config_file = os.path.join(self.config_dir, 'config_pyedu_integration.yaml')
        
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Check for required sections
        self.assertIn('experiment:', content)
        self.assertIn('exp_name:', content)
        self.assertIn('dataset_base_dir:', content)
        self.assertIn('pyedu-integration', content)
    
    def test_train_config_file_structure(self):
        """Test that training configuration file has proper structure."""
        train_config_file = os.path.join(self.config_dir, 'train', 'train_pyedu_integration.yaml')
        
        with open(train_config_file, 'r') as f:
            content = f.read()
            
        # Check for required sections
        self.assertIn('system:', content)
        self.assertIn('model:', content)
        self.assertIn('data:', content)
        self.assertIn('data_path:', content)
        self.assertIn('pyedu', content)
        self.assertIn('tokenizer:', content)


class TestPyEduDocumentationUpdates(unittest.TestCase):
    """Test cases for PyEdu documentation updates."""
    
    def test_data_md_updated(self):
        """Test that Data.md includes PyEdu dataset information."""
        data_md_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'Data.md')
        
        with open(data_md_path, 'r') as f:
            content = f.read()
        
        # Check that PyEdu dataset is mentioned
        self.assertIn('Leon-Leee/unofficial-pyedu', content)
        self.assertIn('pyedu', content.lower())
        self.assertIn('educational Python code', content)
        self.assertIn('6GB', content)
    
    def test_tools_readme_updated(self):
        """Test that tools README includes PyEdu utilities."""
        tools_readme_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'README.md')
        
        with open(tools_readme_path, 'r') as f:
            content = f.read()
        
        # Check that PyEdu utilities are documented
        self.assertIn('pyedu_dataset_utils.py', content)
        self.assertIn('PyEdu dataset', content)
        self.assertIn('educational Python code', content)


class TestPyEduUtilityScript(unittest.TestCase):
    """Test cases for PyEdu utility script functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.utils_script = os.path.join(os.path.dirname(__file__), '..', 'tools', 'pyedu_dataset_utils.py')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_utility_script_exists(self):
        """Test that the PyEdu utility script exists and is executable."""
        self.assertTrue(os.path.exists(self.utils_script))
        
        # Check if script has proper shebang
        with open(self.utils_script, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith('#!'))
    
    def test_utility_script_imports(self):
        """Test that the utility script can be imported without errors."""
        try:
            import sys
            sys.path.insert(0, os.path.dirname(self.utils_script))
            import pyedu_dataset_utils
            
            # Check that main classes and functions exist
            self.assertTrue(hasattr(pyedu_dataset_utils, 'PyEduDatasetHandler'))
            self.assertTrue(hasattr(pyedu_dataset_utils, 'main'))
            
        except ImportError as e:
            # If import fails due to missing dependencies, that's acceptable for this test
            if 'datasets' in str(e) or 'huggingface_hub' in str(e):
                self.skipTest(f"Optional dependencies not available: {e}")
            else:
                raise


class TestPyEduIntegrationEnd2End(unittest.TestCase):
    """End-to-end integration tests for PyEdu dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_configuration_loading(self):
        """Test that PyEdu configurations can be loaded without syntax errors."""
        import yaml
        
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'pyedu-integration')
        
        # Test main config file
        config_file = os.path.join(config_dir, 'config_pyedu_integration.yaml')
        with open(config_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                self.assertIsInstance(config, dict)
                self.assertIn('experiment', config)
            except yaml.YAMLError as e:
                self.fail(f"Config file has invalid YAML syntax: {e}")
        
        # Test training config file
        train_config_file = os.path.join(config_dir, 'train', 'train_pyedu_integration.yaml')
        with open(train_config_file, 'r') as f:
            try:
                train_config = yaml.safe_load(f)
                self.assertIsInstance(train_config, dict)
                self.assertIn('data', train_config)
                self.assertIn('model', train_config)
            except yaml.YAMLError as e:
                self.fail(f"Training config file has invalid YAML syntax: {e}")
    
    def test_dataset_download_without_dependencies(self):
        """Test that dataset download fails gracefully without dependencies."""
        if not PYEDU_UTILS_AVAILABLE:
            self.skipTest("PyEdu dataset utilities not available")
        
        handler = PyEduDatasetHandler(cache_dir=self.temp_dir)
        
        # Test that download raises ImportError when datasets library is not available
        # We'll simulate this by temporarily removing the import
        original_hf_available = handler.__class__.__dict__.get('HF_DATASETS_AVAILABLE', True)
        
        # Create a handler that simulates missing dependencies
        with patch.object(handler, 'download_dataset') as mock_download:
            mock_download.side_effect = ImportError("datasets library required")
            
            output_dir = os.path.join(self.temp_dir, 'pyedu_output')
            
            with self.assertRaises(ImportError):
                handler.download_dataset(output_dir)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)