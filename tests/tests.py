import unittest
from pathlib import Path
import pandas as pd
from key_extractor.key_extractor import (
    KeyExtractor,
    KeyExtractorDataFrame,
    KeyExtractorSeries
)
from textwrap import dedent
import io
import sys
from contextlib import contextmanager

@contextmanager
def capture_stdout():
    """Capture stdout for testing print output."""
    stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

class TestKeyExtractor(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'Z'],
            'value': [1, 2, 3, 4, 5]
        })
        self.series = pd.Series([1, 2, 3, 4, 5], name='values')

    def test_df_basic_extraction(self):
        """Test basic key extraction from DataFrame."""
        result = self.df.askeys('value', to='series')
        expected = pd.Series([1, 2, 3, 4, 5], name='value')
        pd.testing.assert_series_equal(result, expected)

    def test_df_unique_values(self):
        """Test extraction with unique values from DataFrame."""
        df = pd.concat([self.df] * 2)  # Create duplicates
        result = df.askeys('value', unique=True, to='series')
        expected = pd.Series([1, 2, 3, 4, 5], name='value')
        pd.testing.assert_series_equal(result, expected)

    def test_df_groupby_single(self):
        """Test grouping by a single column in DataFrame."""
        result = self.df.askeys('value', groupby='category', to='str')
        expected = dedent("""
            [category: A] (2)
            1;2

            [category: B] (2)
            3;4

            [category: C] (1)
            5

            """).lstrip()
        self.assertEqual(result, expected)

    def test_df_groupby_multiple(self):
        """Test grouping by multiple columns in DataFrame."""
        result = self.df.askeys(
            'value',
            groupby=['category', 'subcategory'],
            to='str'
        )
        expected = dedent("""
            [category: A | subcategory: X] (1)
            1

            [category: A | subcategory: Y] (1)
            2

            [category: B | subcategory: X] (1)
            3

            [category: B | subcategory: Y] (1)
            4

            [category: C | subcategory: Z] (1)
            5

            """).lstrip()
        self.assertEqual(result, expected)

    def test_df_batching(self):
        """Test batch creation in DataFrame."""
        result = self.df.askeys(
            'value',
            batch_size=2,
            to='str'
        )
        expected = dedent("""
            [batch: 1] (2)
            1;2

            [batch: 2] (2)
            3;4

            [batch: 3] (1)
            5

            """).lstrip()
        self.assertEqual(result, expected)

    def test_series_basic_extraction(self):
        """Test basic key extraction from Series."""
        result = self.series.askeys(to='series')
        pd.testing.assert_series_equal(result, self.series)

    def test_series_batching(self):
        """Test batch creation in Series."""
        result = self.series.askeys(batch_size=2, to='str')
        expected = dedent("""
            [batch: 1] (2)
            1;2

            [batch: 2] (2)
            3;4

            [batch: 3] (1)
            5

            """).lstrip()
        self.assertEqual(result, expected)

    def test_df_stdout(self):
        """Test stdout output for DataFrame."""
        with capture_stdout() as output:
            self.df.askeys('value', to='stdout')
            self.assertEqual(output.getvalue().strip(), '1;2;3;4;5')

    def test_series_stdout(self):
        """Test stdout output for Series."""
        with capture_stdout() as output:
            self.series.askeys(to='stdout')
            self.assertEqual(output.getvalue().strip(), '1;2;3;4;5')

    def test_invalid_output_type(self):
        """Test error handling for invalid output type."""
        with self.assertRaises(ValueError):
            self.df.askeys('value', to='invalid')

    def test_missing_key(self):
        """Test error handling for missing key in DataFrame."""
        with self.assertRaises(KeyError):
            self.df.askeys('non_existent', to='series')

    def test_file_output(self):
        """Test file output functionality."""
        # Create temporary directory using unittest's temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Test DataFrame file output
            self.df.askeys('value', to_file=path)
            df_files = list(path.glob('*.txt'))
            self.assertEqual(len(df_files), 1)

            # Test Series file output
            self.series.askeys(to_file=path)
            series_files = list(path.glob('*.txt'))
            self.assertEqual(len(series_files), 2)  # Including previous file

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame({'a': [], 'b': []})
        result = empty_df.askeys('a', to='series')
        self.assertEqual(len(result), 0)

    def test_empty_series(self):
        """Test behavior with empty Series."""
        empty_series = pd.Series([], name='empty')
        result = empty_series.askeys(to='series')
        self.assertEqual(len(result), 0)

    def test_none_values(self):
        """Test handling of None values."""
        df = self.df.copy()
        df.loc[0, 'value'] = None
        result = df.askeys('value', to='str')
        self.assertIn('None', result)

    def test_mixed_types(self):
        """Test handling of mixed types."""
        df = self.df.copy()
        df.loc[0, 'value'] = 'string'
        result = df.askeys('value', to='str')
        self.assertIn('string', result)

    def test_different_separators(self):
        """Test different separator characters."""
        separators = [';', '|', ',']
        for sep in separators:
            result = self.df.askeys('value', sep=sep, to='str')
            self.assertIn(sep, result)

if __name__ == '__main__':
    unittest.main(verbosity=2)
