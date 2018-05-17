# pylint: disable=C0111
# pylint: disable=R0904


from __future__ import absolute_import, unicode_literals
import unittest

from tests._sample_metadata import SampleMetaData
from tests._util import MockMetadataMixin

from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from gutenberg.query import list_supported_metadatas


class TestListSupportedMetadatas(unittest.TestCase):
    def test_has_supported_metadatas(self):
        metadatas = list_supported_metadatas()
        self.assertGreater(len(metadatas), 0)


class TestGetMetadata(MockMetadataMixin, unittest.TestCase):
    def sample_data(self):
        return SampleMetaData.all()

    def _run_get_metadata_for_feature(self, feature):
        for testcase in self.sample_data():
            expected = getattr(testcase, feature)
            actual = get_metadata(feature, testcase.etextno)
            self.assertEqual(
                set(actual),
                set(expected),
                'non-matching {feature} for book {etextno}: '
                'expected={expected} actual={actual}'
                .format(
                    feature=feature,
                    etextno=testcase.etextno,
                    actual=actual,
                    expected=expected))

    def test_get_metadata_title(self):
        self._run_get_metadata_for_feature('title')

    def test_get_metadata_author(self):
        self._run_get_metadata_for_feature('author')

    def test_get_metadata_formaturi(self):
        self._run_get_metadata_for_feature('formaturi')

    def test_get_metadata_rights(self):
        self._run_get_metadata_for_feature('rights')

    def test_get_metadata_subject(self):
        self._run_get_metadata_for_feature('subject')

    def test_get_metadata_language(self):
        self._run_get_metadata_for_feature('language')


class TestGetEtexts(MockMetadataMixin, unittest.TestCase):
    def sample_data(self):
        return SampleMetaData.all()

    def _run_get_etexts_for_feature(self, feature):
        for testcase in self.sample_data():
            for feature_value in getattr(testcase, feature):
                actual = get_etexts(feature, feature_value)
                if testcase.is_phantom:
                    self.assertNotIn(testcase.etextno, actual)
                else:
                    self.assertIn(
                        testcase.etextno,
                        actual,
                        "didn't retrieve {etextno} when querying for books "
                        'that have {feature}="{feature_value}" (got {actual}).'
                        .format(
                            etextno=testcase.etextno,
                            feature=feature,
                            feature_value=feature_value,
                            actual=actual))

    def test_get_etexts_title(self):
        self._run_get_etexts_for_feature('title')

    def test_get_etexts_author(self):
        self._run_get_etexts_for_feature('author')

    def test_get_etexts_formaturi(self):
        self._run_get_etexts_for_feature('formaturi')

    def test_get_etexts_language(self):
        self._run_get_etexts_for_feature('language')

    def test_get_etexts_rights(self):
        self._run_get_etexts_for_feature('rights')

    def test_get_etexts_subject(self):
        self._run_get_etexts_for_feature('subject')


if __name__ == '__main__':
    unittest.main()
