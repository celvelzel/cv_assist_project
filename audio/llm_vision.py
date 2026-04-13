"""
LLM Vision Parser Module
========================
Integrates with Poe API (using OpenAI client) to parse user voice commands
with visual context from camera frames for improved accuracy.

When ASR (Automatic Speech Recognition) produces ambiguous or noisy text,
this module sends the text + multiple camera frames to an LLM (deepseek)
for semantic understanding and object identification.

Flow:
    Whisper ASR: "ana more i´m up phone" → LLM Vision: "phone"
"""

import logging
import json
import base64
import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available, frame encoding disabled")

try:
    import openai
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore
    APIConnectionError = APIError = APITimeoutError = RateLimitError = Exception  # type: ignore
    logger.warning("OpenAI client not available, LLM vision disabled")


class LLMVisionParser:
    """
    Parse voice commands using LLM with visual context.
    
    Attributes:
        model_name: LLM model to use (e.g., "deepseek-v3.2")
        api_key: Poe API key for authentication
        timeout_sec: API call timeout in seconds
        max_frames: Maximum number of frames to send (4 recommended)
        retry_count: Number of retries on API failure
    """
    
    def __init__(self,
                 api_key: str,
                 model_name: str = "deepseek-v3.2",
                 timeout_sec: float = 5.0,
                 max_frames: int = 4,
                 retry_count: int = 1):
        """
        Initialize LLM Vision Parser.
        
        Parameters:
            api_key: Poe API key (from environment or config)
            model_name: Poe model identifier
            timeout_sec: API call timeout (seconds)
            max_frames: Max frames to include in request
            retry_count: Retry attempts on failure
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI client not installed. "
                "Install with: pip install openai"
            )
        
        if not api_key or api_key == "your_poe_api_key_here":
            logger.warning(
                "POE_API_KEY not configured. LLM vision parsing will be disabled. "
                "Set POE_API_KEY in .env or config to enable this feature."
            )
            self.enabled = False
            return
        
        self.enabled = True
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_frames = max(1, min(max_frames, 4))  # Clamp 1-4
        self.retry_count = max(0, retry_count)
        
        try:
            # Initialize Poe client with OpenAI SDK
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.poe.com/v1",
                timeout=timeout_sec
            )
            logger.info(
                f"LLM Vision Parser initialized: "
                f"model={model_name}, timeout={timeout_sec}s, "
                f"max_frames={self.max_frames}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Poe client: {e}")
            self.enabled = False
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """
        Convert a frame (numpy array) to base64 for API transmission.
        
        Parameters:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            Base64-encoded JPEG string, or None on error
        """
        if not CV2_AVAILABLE:
            logger.debug("cv2 not available, cannot encode frame")
            return None
        
        try:
            # Compress to JPEG for smaller payload
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            return None
    
    def _build_vision_prompt(self, asr_text: str, frame_count: int) -> str:
        """
        Build the LLM prompt for object parsing with vision context.
        
        Parameters:
            asr_text: Text from Whisper ASR
            frame_count: Number of frames being provided
            
        Returns:
            Structured prompt for LLM
        """
        prompt = (
            f"You are a helpful assistant for a visual assistance system for the visually impaired. "
            f"The user has spoken the following phrase: \"{asr_text}\"\n\n"
            f"You are provided with {frame_count} consecutive camera frame(s) showing the user's environment.\n\n"
            f"Task: Analyze the speech and the visual context to identify the object the user wants to find.\n\n"
            f"Instructions:\n"
            f"1. Extract the most likely target object from the user's speech.\n"
            f"2. Consider what objects might be visible in the provided frame(s).\n"
            f"3. Return a simple, common English object phrase suitable for object detection and include the indefinite article when natural (e.g., 'a bottle', 'a cup', 'a chair').\n"
            f"4. Ignore filler words, background noise, and spoken artifacts.\n"
            f"5. If the user mentions an object without an article, normalize it to the article form such as 'bottle' -> 'a bottle'.\n"
            f"6. If you cannot determine the object, return the most reasonable guess based on context.\n\n"
            f"Return ONLY a JSON object in this exact format:\n"
            f'{{"target": "<object_phrase>"}}\n\n'
            f"Example responses:\n"
            f'{{"target": "a phone"}}\n'
            f'{{"target": "a cup"}}\n'
            f'{{"target": "a bottle"}}\n\n'
            f"Do not include any other text, only the JSON."
        )
        return prompt
    
    def parse_with_vision(
        self,
        asr_text: str,
        frames: List[np.ndarray],
    ) -> Tuple[Optional[Dict[str, str]], Optional[float], bool]:
        """
        Parse ASR text with visual context using LLM.

        Returns:
            (result_dict_or_None, poe_elapsed_ms_or_None, poe_invoked).
            poe_invoked is True iff the Poe API path was entered (including failed calls).
        """
        if not self.enabled:
            logger.debug("LLM vision parsing disabled, returning None")
            return None, None, False

        if not asr_text or not asr_text.strip():
            logger.warning("Empty ASR text provided to LLM vision parser")
            return None, None, False

        if not frames or len(frames) == 0:
            logger.warning("No frames provided to LLM vision parser")
            return None, None, False

        frames = frames[:self.max_frames]
        frame_count = len(frames)

        logger.debug(f"Parsing with vision: text='{asr_text}', frames={frame_count}")

        try:
            message_content = [
                {
                    "type": "text",
                    "text": self._build_vision_prompt(asr_text, frame_count),
                }
            ]

            for i, frame in enumerate(frames):
                img_base64 = self.encode_frame_to_base64(frame)
                if img_base64:
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64,
                        },
                    })
                    logger.debug(f"Frame {i+1}/{frame_count} encoded")
                else:
                    logger.warning(f"Failed to encode frame {i+1}")

            result, elapsed_ms = self._call_poe_api_with_retry(message_content)
            if result is None:
                logger.warning("LLM API returned None, falling back to ASR text")
            else:
                logger.info(f"LLM parsing result: {result}")
            return result, float(elapsed_ms), True

        except Exception as e:
            logger.error(f"LLM vision parsing exception: {e}", exc_info=True)
            return None, None, False

    def _call_poe_api_with_retry(self, message_content: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Call Poe API with automatic retry and JSON parsing.

        Returns:
            (Parsed JSON response or None, elapsed_ms for entire call including retries)
        """
        t0 = time.perf_counter()
        attempt = 0
        while attempt <= self.retry_count:
            try:
                logger.debug(
                    f"Calling Poe API (attempt {attempt + 1}/{self.retry_count + 1})"
                )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                    temperature=0.3,  # Low temperature for deterministic outputs
                    max_tokens=100     # Keep response short
                )
                
                # Extract response text
                response_text = response.choices[0].message.content.strip()
                logger.debug(f"API response: {response_text}")
                
                # Parse JSON response
                result = self._parse_json_response(response_text)
                if result:
                    return result, (time.perf_counter() - t0) * 1000.0

            except APIConnectionError as e:
                logger.error(f"API connection error: {e}")
                attempt += 1
            except APITimeoutError as e:
                logger.error(f"API timeout: {e}")
                attempt += 1
            except RateLimitError as e:
                logger.error(f"API rate limit exceeded: {e}")
                return None, (time.perf_counter() - t0) * 1000.0
            except APIError as e:
                logger.error(f"API error: {e}")
                attempt += 1
            except Exception as e:
                logger.error(f"Unexpected error calling Poe API: {e}")
                attempt += 1
        
        logger.warning("LLM API call failed after retries")
        return None, (time.perf_counter() - t0) * 1000.0
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, str]]:
        """
        Parse JSON from LLM response.
        
        Parameters:
            response_text: Raw response from LLM
            
        Returns:
            {"target": "object_name"} or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # First, try direct parsing
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if '```' in response_text:
                    # Extract content between ``` markers
                    parts = response_text.split('```')
                    if len(parts) >= 2:
                        json_str = parts[1]
                        # Remove "json" language marker if present
                        if json_str.startswith('json'):
                            json_str = json_str[4:]
                        data = json.loads(json_str.strip())
                    else:
                        logger.error(f"Could not extract JSON from: {response_text}")
                        return None
                else:
                    logger.error(f"Response is not valid JSON: {response_text}")
                    return None
            
            # Validate response structure
            if isinstance(data, dict) and "target" in data:
                target = data["target"].strip()
                if target and len(target) > 0:
                    return {"target": target}
            
            logger.warning(f"Invalid response format: {data}")
            return None
            
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return None


# Convenience function for testing
def create_llm_vision_parser(config_dict: Dict[str, Any]) -> Optional[LLMVisionParser]:
    """
    Factory function to create LLMVisionParser from config.
    
    Parameters:
        config_dict: Config dict with keys: poe_api_key, poe_model, poe_timeout_sec, etc.
        
    Returns:
        LLMVisionParser instance or None if disabled/missing API key
    """
    try:
        api_key = config_dict.get("poe_api_key", "")
        
        if not api_key or api_key == "your_poe_api_key_here":
            logger.debug("LLM vision parsing not configured")
            return None
        
        parser = LLMVisionParser(
            api_key=api_key,
            model_name=config_dict.get("poe_model", "deepseek-v3.2"),
            timeout_sec=config_dict.get("poe_timeout_sec", 5.0),
            max_frames=config_dict.get("max_frames_for_vision", 4),
            retry_count=config_dict.get("api_retry_count", 1)
        )
        
        if parser.enabled:
            return parser
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to create LLM vision parser: {e}")
        return None
