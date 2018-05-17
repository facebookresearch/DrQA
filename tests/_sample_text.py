# pylint: disable=C0111
# pylint: disable=R0903


from __future__ import absolute_import, unicode_literals
from io import open
import os
import sys

from gutenberg._util.os import determine_encoding


class SampleText(object):
    def __init__(self, etextno, raw_text, clean_text):
        self.etextno = etextno
        self.raw_text = raw_text
        self.clean_text = clean_text

    @classmethod
    def for_etextno(cls, etextno):
        raw_text = _load_rawtext(etextno)
        clean_text = _load_cleantext(etextno)
        return SampleText(etextno, raw_text, clean_text)

    @staticmethod
    def all():
        raw_texts = set(os.listdir(_rawtext_data_path()))
        clean_texts = set(os.listdir(_cleantext_data_path()))
        for etextno in raw_texts & clean_texts:
            yield SampleText.for_etextno(int(etextno))


def _rawtext_data_path():
    module = os.path.dirname(sys.modules['tests'].__file__)
    return os.path.join(module, 'data', 'raw-texts')


def _cleantext_data_path():
    module = os.path.dirname(sys.modules['tests'].__file__)
    return os.path.join(module, 'data', 'clean-texts')


def _load_rawtext(etextno):
    data_path = os.path.join(_rawtext_data_path(), str(etextno))
    encoding = determine_encoding(data_path, 'utf-8')
    return open(data_path, encoding=encoding).read()


def _load_cleantext(etextno):
    data_path = os.path.join(_cleantext_data_path(), str(etextno))
    encoding = determine_encoding(data_path, 'utf-8')
    return open(data_path, encoding=encoding).read()
