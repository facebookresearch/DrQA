# pylint: disable=C0111
# pylint: disable=R0904


from __future__ import absolute_import, unicode_literals
import unittest

from gutenberg._domain_model.exceptions import InvalidEtextIdException
from gutenberg._domain_model.types import validate_etextno


class TestValidateEtextno(unittest.TestCase):
    def test_is_valid_etext(self):
        self.assertIsNotNone(validate_etextno(1))
        self.assertIsNotNone(validate_etextno(12))
        self.assertIsNotNone(validate_etextno(123))
        self.assertIsNotNone(validate_etextno(1234))

    def test_is_invalid_etext(self):
        self.assertRaises(InvalidEtextIdException, validate_etextno, 'not-int')
        self.assertRaises(InvalidEtextIdException, validate_etextno, -123)
        self.assertRaises(InvalidEtextIdException, validate_etextno, 0)
        self.assertRaises(InvalidEtextIdException, validate_etextno, 12.3)


if __name__ == '__main__':
    unittest.main()
