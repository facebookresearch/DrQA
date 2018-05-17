# pylint: disable=C0111
# pylint: disable=R0903
# pylint: disable=R0904


from __future__ import absolute_import, unicode_literals
import abc
import codecs
import os
import shutil
import tempfile
import unittest

from six import with_metaclass

from gutenberg._util.abc import abstractclassmethod
from gutenberg._util.objects import all_subclasses
from gutenberg._util.os import makedirs
from gutenberg._util.os import remove
from gutenberg._util.os import reopen_encoded
from tests._util import always_throw


class TestAllSubclasses(unittest.TestCase):
    def test_all_subclasses(self):
        class Root(object):
            pass

        class AB(Root):
            pass

        class ABC(AB):
            pass

        class AD(Root):
            pass

        class ABAD(AB, AD):
            pass

        class ABADE(ABAD):
            pass

        self.assertTrue(all_subclasses(Root), set([AB, ABC, AD, ABAD, ABADE]))
        self.assertEqual(all_subclasses(ABADE), set())


class TestAbstractClassMethod(unittest.TestCase):
    def test_abstractclassmethod(self):
        class ClassWithAbstractClassMethod(
                with_metaclass(abc.ABCMeta, object)):
            @abstractclassmethod
            def method(cls):
                pass

        class ConcreteImplementation(ClassWithAbstractClassMethod):
            @classmethod
            def method(cls):
                pass

        self.assertRaises(TypeError, ClassWithAbstractClassMethod)
        ConcreteImplementation()


class TestRemove(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.temporary_file = tempfile.NamedTemporaryFile(delete=False).name

    def test_remove(self):
        for path in (self.temporary_file, self.temporary_directory):
            self.assertTrue(os.path.exists(path))
            remove(path)
            self.assertFalse(os.path.exists(path))


class TestMakedirs(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)

    def test_makedirs(self):
        path = os.path.join(self.temporary_directory, 'foo', 'bar', 'baz')
        self.assertFalse(os.path.exists(path))
        makedirs(path)
        self.assertTrue(os.path.exists(path))

    def test_makedirs_does_not_swallow_exception(self):
        original_makedirs = os.makedirs
        os.makedirs = always_throw(OSError)
        with self.assertRaises(OSError):
            makedirs('/some/path')
        os.makedirs = original_makedirs


class TestReopenEncoded(unittest.TestCase):
    def setUp(self):
        self.temporary_path = tempfile.mktemp()

    def tearDown(self):
        remove(self.temporary_path)

    def test_reopen_encoded(self):
        for encoding in ('utf-8', 'utf-16'):
            with codecs.open(self.temporary_path, 'w', encoding) as fobj:
                fobj.write('something')

            with open(self.temporary_path, 'r') as fobj:
                reopened_fobj = reopen_encoded(fobj, fobj.mode)
                self.assertEqual(reopened_fobj.encoding.lower(), encoding)


if __name__ == '__main__':
    unittest.main()
