import os
import sys
import unittest
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio.intent_parser import RegexIntentParser, create_intent_parser
from core.interfaces import ParsedIntent


class TestRegexIntentParser(unittest.TestCase):
    """Tests for RegexIntentParser implementation."""
    
    def setUp(self):
        self.parser = RegexIntentParser()
    
    def test_chinese_left_modifier(self):
        """Test parsing Chinese left spatial modifier."""
        test_cases = [
            "找到左边的杯子",
            "左边的杯子",
            "左侧的手机",
            "寻找左边的瓶子",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertEqual(result.spatial_modifier, "left")
                self.assertIn("杯" if "杯" in text else "手" if "手" in text else "瓶", result.target_class)
                self.assertEqual(result.raw_text, text.lower())
    
    def test_chinese_right_modifier(self):
        """Test parsing Chinese right spatial modifier."""
        test_cases = [
            "找到右边的杯子",
            "右边的手机",
            "右侧的瓶子",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertEqual(result.spatial_modifier, "right")
    
    def test_english_left_modifier(self):
        """Test parsing English left spatial modifier."""
        test_cases = [
            "find the left cup",
            "locate the left bottle",
            "the left phone",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertEqual(result.spatial_modifier, "left")
    
    def test_english_right_modifier(self):
        """Test parsing English right spatial modifier."""
        test_cases = [
            "find the right cup",
            "locate the right bottle", 
            "the right phone",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertEqual(result.spatial_modifier, "right")
    
    def test_no_spatial_modifier(self):
        """Test parsing without spatial modifier."""
        test_cases = [
            "找到杯子",
            "寻找手机",
            "where is the cup",
            "find bottle",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertIsNone(result.spatial_modifier)
    
    def test_nearest_modifier(self):
        """Test parsing nearest/closest modifier."""
        test_cases = [
            "找到最近的杯子",
            "locate the nearest cup",
            "the closest bottle",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = self.parser.parse(text)
                self.assertIsNotNone(result, f"Failed to parse: {text}")
                self.assertEqual(result.spatial_modifier, "nearest")
    
    def test_empty_input(self):
        """Test parsing empty or invalid input."""
        invalid_inputs = [
            "",
            "   ",
            None,
        ]
        
        for text in invalid_inputs:
            with self.subTest(text=text):
                if text is None:
                    # Skip None test as parser expects string
                    continue
                result = self.parser.parse(text)
                self.assertIsNone(result, f"Expected None for empty input: '{text}'")
    
    def test_factory_function(self):
        """Test factory function creates correct parser type."""
        parser = create_intent_parser("regex")
        self.assertIsInstance(parser, RegexIntentParser)
        
        with self.assertRaises(ValueError):
            create_intent_parser("invalid_type")


class TestParsedIntentIntegration(unittest.TestCase):
    """Integration tests for intent parsing with spatial selection."""
    
    def test_full_parsing_flow(self):
        """Test complete flow from text to structured intent."""
        parser = RegexIntentParser()
        
        # Simulate ASR output
        asr_outputs = [
            "找到左边的杯子",
            "右边的手机在哪里",
            "locate the nearest bottle",
            "where is the cup",
        ]
        
        expected_results = [
            ("杯子", "left"),
            ("手机", "right"),
            ("bottle", "nearest"),
            ("cup", None),
        ]
        
        for text, (expected_target, expected_spatial) in zip(asr_outputs, expected_results):
            with self.subTest(text=text):
                result = parser.parse(text)
                self.assertIsNotNone(result)
                self.assertEqual(result.target_class, expected_target)
                self.assertEqual(result.spatial_modifier, expected_spatial)
    
    def test_parser_consistency(self):
        """Test that parser produces consistent results for similar inputs."""
        parser = RegexIntentParser()
        
        # These should all produce similar results
        similar_inputs = [
            "找到左边的杯子",
            "左边的杯子",
            "左侧的杯子",
        ]
        
        results = [parser.parse(text) for text in similar_inputs]
        
        # All should parse successfully
        for result in results:
            self.assertIsNotNone(result)
        
        # All should have same spatial modifier and target
        base_result = results[0]
        for result in results[1:]:
            self.assertEqual(result.spatial_modifier, base_result.spatial_modifier)
            self.assertEqual(result.target_class, base_result.target_class)


if __name__ == "__main__":
    unittest.main()
