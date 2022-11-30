import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from unittest import TestCase
import unittest
from images.generate_data.main import _generate_data
import click
from click.testing import CliRunner


class TestGenerateData(TestCase):
    def test_correct(self):
        fake_df = load_diabetes(as_frame=True).frame
        fake_df_path = 'data.csv'
        output_path = 'tests/yyyymmdd'
        os.makedirs(output_path, exist_ok=True)
        fake_df.to_csv(fake_df_path, index=None)

        # runner = CliRunner()
        # result = runner.invoke(_generate_data, [output_path])
        # _generate_data(output_path)

        df_train = pd.read_csv(f'{output_path}/data.csv')
        df_target = pd.read_csv(f'{output_path}/target.csv')

        self.assertEqual(len(df_target), len(df_train))
        self.assertEqual(df_target.shape[1] + df_train.shape[1], fake_df.shape[1])


if __name__ == '__main__':
    unittest.main()
