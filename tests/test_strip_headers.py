# pylint: disable=C0111
# pylint: disable=R0904


from __future__ import absolute_import, unicode_literals
import unittest

from tests._sample_text import SampleText

from gutenberg.cleanup import strip_headers


class TestStripHeaders(unittest.TestCase):
    def test_strip_headers(self):
        for testcase in SampleText.all():
            expected = testcase.clean_text.splitlines()
            actual = strip_headers(testcase.raw_text).splitlines()
            lines = list(zip(actual, expected))
            for i, (actual_line, expected_line) in enumerate(lines, start=1):
                self.assertEqual(
                    actual_line,
                    expected_line,
                    'non-matching lines for etext {etextno}:\n'
                    '{previous_lines}\n'
                    '{lineno_separator}\n'
                    'got "{actual}"\n'
                    'expected "{expected}"\n'
                    '{separator}\n'
                    '{next_lines}'
                    .format(
                        etextno=testcase.etextno,
                        previous_lines=_previous_lines(i, lines, amount=3),
                        next_lines=_next_lines(i, lines, amount=3),
                        actual=actual_line,
                        expected=expected_line,
                        lineno_separator='line {0}:'.format(i).center(80, '-'),
                        separator=''.center(80, '-')))


def _previous_lines(i, lines, amount):
    lower = max(0, i-amount)
    prev_lines = lines[lower:i-1]
    return '\n'.join('line {0}: "{1}"'.format(j, line)
                     for j, (_, line) in enumerate(prev_lines, start=lower))


def _next_lines(i, lines, amount):
    upper = min(len(lines), i+amount+1)
    next_lines = lines[i+1:upper]
    return '\n'.join('line {0}: "{1}"'.format(j, line)
                     for j, (_, line) in enumerate(next_lines, start=i+1))


if __name__ == '__main__':
    unittest.main()
