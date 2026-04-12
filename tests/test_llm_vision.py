"""
Test suite for LLM Vision Parser
=================================

Tests for the LLMVisionParser module that integrates Poe API for 
voice command parsing with visual context.

Tests cover:
- Module import and initialization
- Frame encoding to base64
- Successful API response parsing
- Timeout and fallback behavior
- JSON response validation
- Integration with ASREngine
"""

import pytest
import sys
import os
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio.asr import ASREngine
from audio.llm_vision import LLMVisionParser, create_llm_vision_parser

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TestLLMVisionParser:
    """Test LLMVisionParser class"""
    
    def test_init_without_api_key(self):
        """Test initialization without API key (should disable)"""
        parser = LLMVisionParser(
            api_key="",
            model_name="deepseek-v3.2"
        )
        assert parser.enabled == False
    
    def test_init_with_placeholder_api_key(self):
        """Test initialization with placeholder API key (should disable)"""
        parser = LLMVisionParser(
            api_key="your_poe_api_key_here",
            model_name="deepseek-v3.2"
        )
        assert parser.enabled == False
    
    @patch('audio.llm_vision.openai')
    def test_init_with_valid_api_key(self, mock_openai):
        """Test initialization with valid API key"""
        parser = LLMVisionParser(
            api_key="test_key_12345",
            model_name="deepseek-v3.2",
            timeout_sec=5.0,
            max_frames=4
        )
        
        # Should have initialized client
        assert mock_openai.OpenAI.called
        assert parser.enabled == True
        assert parser.model_name == "deepseek-v3.2"
        assert parser.timeout_sec == 5.0
        assert parser.max_frames == 4
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_encode_frame_to_base64(self):
        """Test frame encoding to base64"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        encoded = parser.encode_frame_to_base64(frame)
        
        assert encoded is not None
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_parse_json_response_valid(self):
        """Test parsing valid JSON response"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        response = '{"target": "phone"}'
        result = parser._parse_json_response(response)
        
        assert result is not None
        assert result['target'] == 'phone'
    
    def test_parse_json_response_with_markdown_blocks(self):
        """Test parsing JSON from markdown code blocks"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        response = '```json\n{"target": "cup"}\n```'
        result = parser._parse_json_response(response)
        
        assert result is not None
        assert result['target'] == 'cup'
    
    def test_parse_json_response_invalid_format(self):
        """Test parsing invalid JSON response"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        response = '{"wrong_key": "value"}'
        result = parser._parse_json_response(response)
        
        assert result is None
    
    def test_parse_json_response_empty_target(self):
        """Test parsing JSON with empty target"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        response = '{"target": ""}'
        result = parser._parse_json_response(response)
        
        assert result is None
    
    def test_parse_with_vision_disabled(self):
        """Test parse_with_vision when parser is disabled"""
        parser = LLMVisionParser(
            api_key="",  # Will disable parser
            model_name="deepseek-v3.2"
        )
        
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result, poe_ms, invoked = parser.parse_with_vision("find a phone", frames)

        assert result is None
        assert poe_ms is None
        assert invoked is False

    def test_parse_with_vision_empty_text(self):
        """Test parse_with_vision with empty text"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result, poe_ms, invoked = parser.parse_with_vision("", frames)

        assert result is None
        assert poe_ms is None
        assert invoked is False

    def test_parse_with_vision_no_frames(self):
        """Test parse_with_vision with no frames"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        result, poe_ms, invoked = parser.parse_with_vision("find a phone", None)

        assert result is None
        assert poe_ms is None
        assert invoked is False

    @patch('audio.llm_vision.openai')
    def test_call_poe_api_with_retry_success(self, mock_openai):
        """Test successful Poe API call"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"target": "phone"}'
        parser.client.chat.completions.create.return_value = mock_response
        
        message_content = [{"type": "text", "text": "test prompt"}]
        result, elapsed_ms = parser._call_poe_api_with_retry(message_content)

        assert result is not None
        assert result['target'] == 'phone'
        assert elapsed_ms >= 0.0
    
    @patch('audio.llm_vision.openai')
    def test_call_poe_api_with_retry_timeout(self, mock_openai):
        """Test API timeout with fallback"""
        import openai
        
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2",
            retry_count=1
        )
        
        # Mock timeout error
        parser.client.chat.completions.create.side_effect = openai.APITimeoutError("Timeout")
        
        message_content = [{"type": "text", "text": "test prompt"}]
        result, elapsed_ms = parser._call_poe_api_with_retry(message_content)

        assert result is None
        assert elapsed_ms >= 0.0


class TestLLMVisionIntegration:
    """Test integration with ASREngine"""
    
    def test_asr_parse_command_with_vision_no_llm(self):
        """Test ASREngine.parse_command_with_vision without LLM"""
        # Skip Whisper initialization by mocking
        with patch('audio.asr.whisper'):
            asr = ASREngine(model_name="base")
            frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
            
            target, poe_ms, invoked = asr.parse_command_with_vision(
                "找到手机",
                frames=frames,
                llm_parser=None
            )

            assert target is not None
            assert poe_ms is None
            assert invoked is False
    
    def test_create_llm_vision_parser_no_key(self):
        """Test factory function without API key"""
        config_dict = {
            "poe_api_key": "",
            "poe_model": "deepseek-v3.2"
        }
        
        parser = create_llm_vision_parser(config_dict)
        assert parser is None
    
    def test_create_llm_vision_parser_placeholder_key(self):
        """Test factory function with placeholder API key"""
        config_dict = {
            "poe_api_key": "your_poe_api_key_here",
            "poe_model": "deepseek-v3.2"
        }
        
        parser = create_llm_vision_parser(config_dict)
        assert parser is None
    
    @patch('audio.llm_vision.openai')
    def test_create_llm_vision_parser_valid_key(self, mock_openai):
        """Test factory function with valid API key"""
        config_dict = {
            "poe_api_key": "test_key_12345",
            "poe_model": "deepseek-v3.2",
            "poe_timeout_sec": 5.0,
            "max_frames_for_vision": 4,
            "api_retry_count": 1
        }
        
        parser = create_llm_vision_parser(config_dict)
        
        assert parser is not None
        assert parser.enabled == True
        assert parser.timeout_sec == 5.0


class TestLLMVisionPrompt:
    """Test prompt engineering"""
    
    def test_build_vision_prompt(self):
        """Test prompt generation"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        prompt = parser._build_vision_prompt("find a phone", 2)
        
        assert "find a phone" in prompt
        assert "2 consecutive camera frame" in prompt
        assert '{"target":' in prompt
        assert "JSON" in prompt
        assert '"a bottle"' in prompt
        assert "include the indefinite article" in prompt


class TestFrameEncoding:
    """Test frame encoding edge cases"""
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_encode_various_frame_sizes(self):
        """Test encoding frames of various sizes"""
        parser = LLMVisionParser(
            api_key="test_key",
            model_name="deepseek-v3.2"
        )
        
        sizes = [(480, 640), (720, 1280), (360, 480)]
        
        for h, w in sizes:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            encoded = parser.encode_frame_to_base64(frame)
            
            assert encoded is not None
            assert isinstance(encoded, str)
    
    def test_encode_frame_without_cv2(self):
        """Test encoding without cv2 (should return None)"""
        with patch('audio.llm_vision.CV2_AVAILABLE', False):
            parser = LLMVisionParser(
                api_key="test_key",
                model_name="deepseek-v3.2"
            )
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = parser.encode_frame_to_base64(frame)
            
            # Should fail gracefully
            assert result is None


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
