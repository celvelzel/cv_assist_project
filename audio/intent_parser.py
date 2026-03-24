# audio/intent_parser.py
"""
Intent parsing module for extracting structured commands from ASR text.
Supports extensible parser implementations via Protocol interfaces.
"""

import re
import logging
from typing import Optional, List

from core.interfaces import ParsedIntent, IIntentParser

logger = logging.getLogger(__name__)


class RegexIntentParser:
    """
    Rule-based intent parser using regex pattern matching.
    Extracts spatial modifiers and target objects from natural language.
    """
    
    def __init__(self):
        # Chinese spatial modifiers
        self.zh_spatial = {
            '左边': 'left',
            '左侧': 'left', 
            '左边的': 'left',
            '左侧的': 'left',
            '右边': 'right',
            '右侧': 'right',
            '右边的': 'right',
            '右侧的': 'right',
            '中间': 'center',
            '中间的': 'center',
            'nearest': 'nearest',
            '最近的': 'nearest',
        }
        
        # English spatial modifiers
        self.en_spatial = {
            'left': 'left',
            'leftmost': 'left',
            'on the left': 'left',
            'right': 'right',
            'rightmost': 'right',
            'on the right': 'right',
            'center': 'center',
            'middle': 'center',
            'nearest': 'nearest',
            'closest': 'nearest',
        }
        
        # Chinese command patterns
        self.zh_patterns = [
            r'找到?(.{1,2}?)的(.+)',
            r'寻找(.{1,2}?)的(.+)',
            r'搜索(.{1,2}?)的(.+)',
            r'定位(.{1,2}?)的(.+)',
            r'(.{1,2}?)的(.+)',
        ]
        
        # English command patterns  
        self.en_patterns = [
            r'find the (\w+) (\w+)',
            r'locate the (\w+) (\w+)',
            r'where is the (\w+) (\w+)',
            r'the (\w+) (\w+)',
            r'find (\w+) (\w+)',
        ]
        
        # Build combined spatial mapping
        self.all_spatial = {**self.zh_spatial, **self.en_spatial}
    
    def parse(self, text: str) -> Optional[ParsedIntent]:
        """
        Parse raw ASR text into structured intent.
        
        Args:
            text: Raw speech transcription (e.g., "找到左边的杯子")
            
        Returns:
            ParsedIntent with target class and spatial modifier, or None if unparseable.
        """
        if not text or not text.strip():
            return None
        
        text = text.strip().lower()
        logger.debug(f"Parsing intent from: '{text}'")
        
        # Try Chinese patterns first
        result = self._parse_chinese(text)
        if result:
            return result
        
        # Try English patterns
        result = self._parse_english(text)
        if result:
            return result
        
        # Try simple spatial modifier extraction
        result = self._extract_simple_spatial(text)
        if result:
            return result
        
        # Fallback: treat entire text as target with no spatial modifier
        logger.warning(f"Could not parse spatial modifier from: '{text}', using as target only")
        return ParsedIntent(
            target_class=text,
            spatial_modifier=None,
            raw_text=text
        )
    
    def _parse_chinese(self, text: str) -> Optional[ParsedIntent]:
        """Parse Chinese spatial commands."""
        # Sort modifiers by length (longest first) to match most specific
        sorted_modifiers = sorted(self.zh_spatial.keys(), key=len, reverse=True)
        
        for modifier_text in sorted_modifiers:
            if modifier_text in text:
                # Extract target after modifier
                parts = text.split(modifier_text, 1)
                if len(parts) > 1:
                    target = parts[1].strip()
                    # Remove common fillers and question words
                    for filler in ['的', '一下', '我的', '这个', '那个', '在哪里', '在哪里呢', '呢', '吗']:
                        target = target.replace(filler, '').strip()
                    
                    # Handle "找到[modifier]的[object]" pattern
                    if '找到' in target:
                        target = target.replace('找到', '').strip()
                    
                    if target:
                        spatial = self.zh_spatial[modifier_text]
                        logger.debug(f"Chinese parse: modifier='{modifier_text}' -> '{spatial}', target='{target}'")
                        return ParsedIntent(
                            target_class=target,
                            spatial_modifier=spatial,
                            raw_text=text
                        )
        
        # Handle "找到[object]" pattern without spatial modifier
        if '找到' in text:
            target = text.replace('找到', '').strip()
            # Remove common fillers
            for filler in ['的', '一下', '我的', '这个', '那个', '在哪里', '在哪里呢', '呢', '吗']:
                target = target.replace(filler, '').strip()
            
            if target:
                logger.debug(f"Chinese parse: '找到' pattern, target='{target}'")
                return ParsedIntent(
                    target_class=target,
                    spatial_modifier=None,
                    raw_text=text
                )
        
        return None
    
    def _parse_english(self, text: str) -> Optional[ParsedIntent]:
        """Parse English spatial commands."""
        words = text.split()
        
        # Check for spatial modifier at different positions
        spatial_modifiers = ['left', 'right', 'center', 'middle', 'nearest', 'closest']
        
        for modifier in spatial_modifiers:
            if modifier in words:
                idx = words.index(modifier)
                
                # Check if modifier is followed by an object
                if idx + 1 < len(words):
                    target = words[idx + 1]
                    # Remove common articles
                    target = target.replace('the', '').replace('a', '').replace('an', '').strip()
                    
                    if target:
                        spatial = self.en_spatial.get(modifier, modifier)
                        logger.debug(f"English parse: modifier='{modifier}' -> '{spatial}', target='{target}'")
                        return ParsedIntent(
                            target_class=target,
                            spatial_modifier=spatial,
                            raw_text=text
                        )
        
        # Handle "where is the [object]" pattern
        if "where is" in text:
            # Extract object after "where is the"
            pattern = r'where is (?:the|a|an)?\s*(\w+)'
            match = re.search(pattern, text)
            if match:
                target = match.group(1)
                logger.debug(f"English parse: 'where is' pattern, target='{target}'")
                return ParsedIntent(
                    target_class=target,
                    spatial_modifier=None,
                    raw_text=text
                )
        
        # Handle "find the [object]" pattern  
        if "find" in words:
            idx = words.index("find")
            if idx + 1 < len(words):
                target = words[idx + 1]
                # Skip articles
                if target in ['the', 'a', 'an'] and idx + 2 < len(words):
                    target = words[idx + 2]
                
                target = target.replace('the', '').replace('a', '').replace('an', '').strip()
                if target:
                    logger.debug(f"English parse: 'find' pattern, target='{target}'")
                    return ParsedIntent(
                        target_class=target,
                        spatial_modifier=None,
                        raw_text=text
                    )
        
        return None
    
    def _extract_simple_spatial(self, text: str) -> Optional[ParsedIntent]:
        """Extract spatial modifier from simple text patterns."""
        # Look for any spatial term in the text
        for modifier_text, spatial in self.all_spatial.items():
            if modifier_text in text:
                # Try to extract target after the modifier
                parts = text.split(modifier_text, 1)
                if len(parts) > 1:
                    target = parts[1].strip()
                    # Remove common fillers
                    for filler in ['的', '一下', '我的', '这个', '那个', 'the', 'a', 'an']:
                        target = target.replace(filler, '').strip()
                    
                    if target and len(target) > 0:
                        logger.debug(f"Simple spatial: modifier='{modifier_text}' -> '{spatial}', target='{target}'")
                        return ParsedIntent(
                            target_class=target,
                            spatial_modifier=spatial,
                            raw_text=text
                        )
        
        return None


# Factory function for creating parsers
def create_intent_parser(parser_type: str = "regex") -> IIntentParser:
    """
    Factory function to create intent parser instances.
    
    Args:
        parser_type: Type of parser to create ("regex" or future "llm")
        
    Returns:
        An implementation of IIntentParser
    """
    if parser_type == "regex":
        return RegexIntentParser()
    # Future: add LLMIntentParser here
    # elif parser_type == "llm":
    #     return LLMIntentParser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
