import unittest
from dataset_validation import normalize, exact_match, token_f1

class MetricTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize('The, CAT!'), 'cat')
    def test_exact_match(self):
        self.assertEqual(exact_match('The Paris', 'paris'), 1)
        self.assertEqual(exact_match('London', 'Paris'), 0)
    def test_f1(self):
        self.assertEqual(token_f1('red blue', 'red green'), 0.5)
        self.assertEqual(token_f1('', ''), 1.0)
        self.assertEqual(token_f1('a', 'elephant'), 0.0)

if __name__ == '__main__':
    unittest.main()
