import unittest

from services.date_parser import format_date_for_display, merge_date_range, parse_date_text


class DateParserTest(unittest.TestCase):
    def test_formats_single_day_for_display(self):
        self.assertEqual(format_date_for_display("7.10"), "7\u670810\u65e5")

    def test_formats_same_month_range_for_display(self):
        self.assertEqual(format_date_for_display("7.20-21"), "7\u670820\u65e5-21\u65e5")

    def test_formats_cross_month_range_for_display(self):
        self.assertEqual(format_date_for_display("10.31-11.1"), "10\u670831\u65e5-11\u67081\u65e5")

    def test_merges_adjacent_column_end_day(self):
        self.assertEqual(merge_date_range("7 / 20", "-21"), "7.20-21")

    def test_merges_dot_prefixed_end_day_from_ocr(self):
        self.assertEqual(merge_date_range("9 / 25", "\u00b7 26"), "9.25-26")

    def test_merges_full_cross_month_range(self):
        self.assertEqual(merge_date_range("10 / 31", "11 / 1"), "10.31-11.1")

    def test_parse_date_text_ignores_spaces(self):
        self.assertEqual(parse_date_text("1 2 / 26"), (12, 26))


if __name__ == "__main__":
    unittest.main()
