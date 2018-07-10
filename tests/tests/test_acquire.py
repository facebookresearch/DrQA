# -*- coding: utf-8 -*-
# pylint: disable=C0111
# pylint: disable=R0904


from __future__ import absolute_import, unicode_literals
from builtins import str
from collections import namedtuple
import itertools
import unittest

from gutenberg._domain_model.exceptions import UnknownDownloadUriException
from gutenberg._domain_model.vocabulary import DCTERMS
from gutenberg._domain_model.vocabulary import PGTERMS
from tests._sample_metadata import SampleMetaData
from tests._util import INTEGRATION_TESTS_ENABLED
from tests._util import MockMetadataMixin
from tests._util import MockTextMixin

from gutenberg.acquire import text
from gutenberg.acquire import load_etext
from gutenberg.acquire import load_metadata


class TestLoadMetadata(MockMetadataMixin, unittest.TestCase):
    def sample_data(self):
        return SampleMetaData.all()

    def test_load_metadata(self):
        metadata = load_metadata()
        self.assertGreater(len(list(metadata[::PGTERMS.ebook])), 0)
        self.assertGreater(len(list(metadata[:DCTERMS.creator:])), 0)
        self.assertGreater(len(list(metadata[:DCTERMS.title:])), 0)


class TestLoadEtext(MockTextMixin, unittest.TestCase):
    def test_load_etext(self):
        loaders = (lambda etextno: load_etext(etextno, refresh_cache=True),
                   lambda etextno: load_etext(etextno, refresh_cache=False))
        testcases = (
            SampleMetaData.for_etextno(2701),   # newstyle identifier
            SampleMetaData.for_etextno(5),      # oldstyle identifier
            SampleMetaData.for_etextno(14287),  # unicode text
            SampleMetaData.for_etextno(23962)   # UTF-8 text
        )
        for testcase, loader in itertools.product(testcases, loaders):
            etext = loader(testcase.etextno)
            self.assertIsInstance(etext, str)
            self.assertNotIn(u'\ufffd', etext)

    def test_invalid_etext(self):
        with self.assertRaises(UnknownDownloadUriException):
            text.load_etext(1, mirror='http://example.com')


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, reason='unit-tests only')
class TestLoadEtextNetworked(unittest.TestCase):
    def test_load_etext(self):
        etext = text.load_etext(2701)
        self.assertIsInstance(etext, str)
        self.assertGreater(len(etext), 1000)


class TestFailLoadEtext(unittest.TestCase):
    def setUp(self):
        self._original_head = text.requests.head

    def tearDown(self):
        text.requests.head = self._original_head

    def request_head_response(self, ok=False):
        response = namedtuple('Response', 'ok')

        def head(*args, **kwargs):
            return response(ok)
        text.requests.head = head

    def test_unreachable_mirror(self):
        self.request_head_response(ok=False)
        with self.assertRaises(UnknownDownloadUriException):
            text.load_etext(1)


if __name__ == '__main__':
    unittest.main()
