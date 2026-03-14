"""
Nova Splitter - 智能消息分段插件 v1.1.0
作者: Nova for 辉宝主人
功能: 
  1. 按字数均分分段（强制标点边界，绝不在词语中间断开）
  2. 按标点分段
  3. LLM辅助智能分段（带超时保护和完善的异常处理）
  4. 智能标点清理(边界清理/句内保留)
"""

import re
import math
import random
import asyncio
import uuid
import traceback
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse, Provider
from astrbot.api.message_components import Plain, BaseMessageComponent, Reply


@dataclass
class SplitResult:
    """分段结果"""
    segments: List[str]
    split_points: List[int]


class SplitStrategy(ABC):
    """分段策略基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.pair_map = {
            '\u201c': '\u201d', '\u300a': '\u300b', '\uff08': '\uff09', '(': ')',
            '[': ']', '{': '}', "'": "'", '\u3010': '\u3011', '<': '>'
        }
        self.quote_chars = {'\u201c', "'", '`'}
    
    @abstractmethod
    def split(self, text: str) -> SplitResult:
        pass
    
    def is_inside_pair(self, text: str, pos: int) -> bool:
        """检查位置是否在配对符号内部"""
        stack = []
        for i, char in enumerate(text[:pos]):
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
            elif char in self.pair_map:
                stack.append(char)
            elif stack:
                expected = self.pair_map.get(stack[-1])
                if char == expected:
                    stack.pop()
        return len(stack) > 0
    
    def find_best_split_point(self, text: str, ideal_pos: int, search_range: int = 30) -> int:
        """在理想位置附近找到最佳分割点
        
        核心原则: 绝不在词语中间断开，必须在标点/空格处切分
        搜索范围扩大到 +-30 字符，优先找句末标点
        """
        # 搜索范围
        start = max(0, ideal_pos - search_range)
        end = min(len(text), ideal_pos + search_range)
        
        best_pos = -1
        best_score = -999
        
        for pos in range(start, end):
            if pos >= len(text):
                continue
            
            char = text[pos]
            score = -999  # 默认不可用
            
            # 只在标点或空格处才是合法切分点
            if char in '\u3002\uff01\uff1f!?\n':
                score = 100  # 句末标点最优
            elif char in '\uff1b;':
                score = 80
            elif char in '\uff0c,\u3001':
                score = 60   # 逗号次之
            elif char in ' ':
                score = 40
            elif char in '\u2026\u2014\u2013~\uff5e':
                score = 35   # 省略号/破折号
            else:
                continue  # 非标点/空格位置直接跳过，绝不在此切分
            
            # 位置越接近理想位置，加分越多
            distance_penalty = abs(pos - ideal_pos) * 2
            score -= distance_penalty
            
            # 如果在配对符号内，大幅减分
            if self.is_inside_pair(text, pos):
                score -= 200
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        # 如果在扩大范围内都没找到标点，进一步扩大搜索
        if best_pos == -1:
            extended_range = min(len(text) // 2, search_range * 3)
            ext_start = max(0, ideal_pos - extended_range)
            ext_end = min(len(text), ideal_pos + extended_range)
            
            for pos in range(ext_start, ext_end):
                if pos >= len(text):
                    continue
                char = text[pos]
                if char in '\u3002\uff01\uff1f!?\n\uff1b;\uff0c,\u3001 ':
                    score = 50 - abs(pos - ideal_pos)
                    if self.is_inside_pair(text, pos):
                        score -= 200
                    if score > best_score:
                        best_score = score
                        best_pos = pos
        
        # 最终兜底: 如果整篇文本都没有标点，才在理想位置切分
        if best_pos == -1:
            best_pos = ideal_pos
        
        return best_pos


class CharCountSplitter(SplitStrategy):
    """按字数均分策略（强制标点边界）"""
    
    def split(self, text: str) -> SplitResult:
        target_segments = self.config.get("target_segments", 3)
        min_segment_length = self.config.get("min_segment_length", 10)
        
        total_length = len(text)
        
        if total_length < min_segment_length * 2:
            return SplitResult(segments=[text], split_points=[])
        
        ideal_length = total_length // target_segments
        
        while ideal_length < min_segment_length and target_segments > 1:
            target_segments -= 1
            ideal_length = total_length // target_segments
        
        if target_segments == 1:
            return SplitResult(segments=[text], split_points=[])
        
        # 计算理想分割点，在标点处切分
        split_points = []
        for i in range(1, target_segments):
            ideal_pos = i * ideal_length
            best_pos = self.find_best_split_point(text, ideal_pos)
            split_points.append(best_pos)
        
        # 去重并排序
        split_points = sorted(set(split_points))
        
        # 移除太靠近开头或结尾的分割点
        split_points = [p for p in split_points 
                       if p >= min_segment_length and p < total_length - min_segment_length]
        
        if not split_points:
            return SplitResult(segments=[text], split_points=[])
        
        # 根据分割点切分文本
        segments = []
        prev_pos = 0
        for pos in split_points:
            if pos > prev_pos:
                segments.append(text[prev_pos:pos + 1])
                prev_pos = pos + 1
        
        if prev_pos < len(text):
            segments.append(text[prev_pos:])
        
        return SplitResult(segments=segments, split_points=split_points)

class PunctuationSplitter(SplitStrategy):
    """按标点分段策略"""
    
    def split(self, text: str) -> SplitResult:
        split_chars = self.config.get("split_punctuation", "\u3002\uff01\uff1f!?\uff1b;\n")
        max_segments = self.config.get("max_segments", 7)
        enable_smart = self.config.get("enable_smart_split", True)
        
        pattern = f"[{re.escape(split_chars)}]+"
        
        if enable_smart:
            segments = self._smart_split(text, pattern)
        else:
            segments = self._simple_split(text, pattern)
        
        if len(segments) > max_segments and max_segments > 0:
            final_segments = segments[:max_segments - 1]
            merged = "".join(segments[max_segments - 1:])
            final_segments.append(merged)
            segments = final_segments
        
        split_points = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(seg)
            split_points.append(pos - 1)
        
        return SplitResult(segments=segments, split_points=split_points)
    
    def _simple_split(self, text: str, pattern: str) -> List[str]:
        parts = re.split(f"({pattern})", text)
        segments = []
        current = ""
        
        for part in parts:
            if not part:
                continue
            if re.fullmatch(pattern, part):
                current += part
                if current.strip():
                    segments.append(current)
                current = ""
            else:
                current += part
        
        if current.strip():
            segments.append(current)
        
        return segments
    
    def _smart_split(self, text: str, pattern: str) -> List[str]:
        compiled = re.compile(pattern)
        stack = []
        segments = []
        current = ""
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
                current += char
                i += 1
                continue
            
            if stack:
                expected = self.pair_map.get(stack[-1])
                if char == expected:
                    stack.pop()
                elif char in self.pair_map:
                    stack.append(char)
                current += char
                i += 1
                continue
            
            if char in self.pair_map:
                stack.append(char)
                current += char
                i += 1
                continue
            
            match = compiled.match(text, pos=i)
            if match:
                delimiter = match.group()
                current += delimiter
                if current.strip():
                    segments.append(current)
                current = ""
                i += len(delimiter)
            else:
                current += char
                i += 1
        
        if current.strip():
            segments.append(current)
        
        return segments


class LLMAssistSplitter(SplitStrategy):
    """LLM辅助分段策略（带超时保护和完善异常处理）"""
    
    def __init__(self, config: dict, context: Context, event_origin: str = None,
                 live_config: AstrBotConfig = None):
        super().__init__(config)
        self.context = context
        self.event_origin = event_origin
        self.live_config = live_config
    
    async def split_async(self, text: str) -> SplitResult:
        """异步分段（需要调用LLM），带超时保护，支持system_prompt"""
        target_segments = self.config.get("target_segments", 3)
        
        if self.live_config:
            provider_id = self.live_config.get("llm_assist_provider_id", "")
        else:
            provider_id = self.config.get("llm_assist_provider_id", "")
        
        # 获取system prompt（分段规则）和 user prompt模板
        system_prompt = self.config.get("llm_split_system_prompt", "")
        user_template = self.config.get("llm_split_prompt",
            "\u539f\u6587\uff1a\n{text}")
        
        # 超时时间可配置
        llm_timeout = float(self.config.get("llm_timeout", 30.0))
        
        logger.info(f"[Nova-Splitter] \u8bfb\u53d6\u5230\u7684provider_id: '{provider_id}'")
        
        # 获取Provider
        provider = None
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning(f"[Nova-Splitter] \u672a\u627e\u5230\u6307\u5b9aProvider: {provider_id}")
        
        if not provider:
            logger.info(f"[Nova-Splitter] \u4f7f\u7528\u4f1a\u8bdd\u9ed8\u8ba4Provider, origin={self.event_origin}")
            provider = self.context.get_using_provider(self.event_origin)
        
        if not provider or not isinstance(provider, Provider):
            logger.warning("[Nova-Splitter] \u672a\u627e\u5230LLM Provider\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5")
            fallback = CharCountSplitter(self.config)
            return fallback.split(text)
        
        user_prompt = user_template.format(n=target_segments, text=text)
        
        try:
            logger.info(f"[Nova-Splitter] \u5f00\u59cb\u8c03\u7528LLM\u5206\u6bb5 (provider: {provider_id or 'default'}, timeout: {llm_timeout}s, system_prompt: {'yes' if system_prompt else 'no'})")
            
            # 构建调用参数
            call_kwargs = {
                "prompt": user_prompt,
                "session_id": uuid.uuid4().hex,
            }
            # 如果配置了system_prompt，使用system_prompt参数
            if system_prompt:
                call_kwargs["system_prompt"] = system_prompt
            
            response = await asyncio.wait_for(
                provider.text_chat(**call_kwargs),
                timeout=llm_timeout
            )
            
            # 安全获取completion_text
            completion_text = None
            if response:
                try:
                    completion_text = response.completion_text
                except Exception as ct_err:
                    logger.error(f"[Nova-Splitter] \u83b7\u53d6completion_text\u5931\u8d25: {ct_err}")
            
            if not completion_text:
                logger.warning(f"[Nova-Splitter] LLM\u8fd4\u56de\u7a7a\u5185\u5bb9\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5")
                fallback = CharCountSplitter(self.config)
                return fallback.split(text)
            
            result_text = completion_text.strip()
            logger.info(f"[Nova-Splitter] LLM\u8fd4\u56de\u5185\u5bb9: {result_text[:100]}...")
            
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                import json
                segments = json.loads(json_match.group())
                if isinstance(segments, list) and len(segments) > 0 and all(isinstance(s, str) for s in segments):
                    logger.info(f"[Nova-Splitter] LLM\u5206\u6bb5\u6210\u529f: {len(segments)} \u6bb5")
                    return SplitResult(segments=segments, split_points=[])
            
            logger.warning(f"[Nova-Splitter] LLM\u8fd4\u56de\u683c\u5f0f\u5f02\u5e38\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5: {result_text[:100]}")
            
        except asyncio.TimeoutError:
            logger.error(f"[Nova-Splitter] LLM\u5206\u6bb5\u8d85\u65f6(15s)\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5")
        except Exception as e:
            logger.error(f"[Nova-Splitter] LLM\u5206\u6bb5\u5931\u8d25: {type(e).__name__}: {e}")
            logger.debug(f"[Nova-Splitter] LLM\u5206\u6bb5\u5f02\u5e38\u8be6\u60c5:\n{traceback.format_exc()}")
        
        # 回退到按字数分段
        logger.info(f"[Nova-Splitter] \u56de\u9000\u4f7f\u7528\u5b57\u6570\u5206\u6bb5")
        fallback = CharCountSplitter(self.config)
        return fallback.split(text)
    
    def split(self, text: str) -> SplitResult:
        fallback = CharCountSplitter(self.config)
        return fallback.split(text)


class PunctuationProcessor:
    """标点处理引擎"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("punctuation_clean_mode", "edge_only")
        self.edge_chars = config.get("edge_clean_chars", "\uff0c,\u3001")
        self.space_chars = config.get("space_replace_chars", "")
        self.custom_regex = config.get("custom_clean_regex", "")
    
    def process(self, segments: List[str], split_points: List[int]) -> List[str]:
        if self.mode == "preserve":
            return segments
        
        processed = []
        for i, seg in enumerate(segments):
            if self.mode == "edge_only":
                seg = self._clean_edge(seg, is_last=(i == len(segments) - 1))
            elif self.mode == "replace_space":
                seg = self._replace_with_space(seg)
            elif self.mode == "custom":
                seg = self._custom_clean(seg)
            processed.append(seg)
        
        return processed
    
    def _clean_edge(self, text: str, is_last: bool = False) -> str:
        if not text:
            return text
        
        while text and text[0] in self.edge_chars:
            text = text[1:]
        
        if not is_last:
            while text and text[-1] in self.edge_chars:
                text = text[:-1]
        
        return text
    
    def _replace_with_space(self, text: str) -> str:
        if not self.space_chars:
            return text
        pattern = f"[{re.escape(self.space_chars)}]"
        return re.sub(pattern, " ", text)
    
    def _custom_clean(self, text: str) -> str:
        if not self.custom_regex:
            return text
        try:
            return re.sub(self.custom_regex, "", text)
        except re.error as e:
            logger.error(f"[Nova-Splitter] \u81ea\u5b9a\u4e49\u6b63\u5219\u9519\u8bef: {e}")
            return text

@register("nova-splitter", "Nova", "智能消息分段插件 - 支持按字数/标点/LLM分段，智能标点清理", "1.1.0")
class NovaSplitterPlugin(Star):
    """Nova智能分段插件 v1.1.0"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.punctuation_processor = PunctuationProcessor(dict(config))
        logger.info("[Nova-Splitter] 插件已初始化 v1.1.0")
        logger.info(f"[Nova-Splitter] 分段模式: {config.get('split_mode', 'char_count')}")
        logger.info(f"[Nova-Splitter] 标点清理模式: {config.get('punctuation_clean_mode', 'edge_only')}")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """标记LLM响应"""
        setattr(event, "__is_llm_reply", True)
        logger.info(f"[Nova-Splitter] 检测到LLM响应，已标记事件")
    
    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """处理消息分段 - 带完整异常保护"""
        logger.info(f"[Nova-Splitter] on_decorating_result 触发")
        
        try:
            await self._do_split(event)
        except Exception as e:
            logger.error(f"[Nova-Splitter] on_decorating_result 发生未预期异常: {type(e).__name__}: {e}")
            logger.error(f"[Nova-Splitter] 异常详情:\n{traceback.format_exc()}")
            # 异常时不修改 result，让框架正常发送原文
    
    async def _do_split(self, event: AstrMessageEvent):
        """实际的分段处理逻辑"""
        # 防重入检查
        if getattr(event, "__nova_splitter_processed", False):
            logger.info(f"[Nova-Splitter] 已处理过，跳过")
            return
        
        result = event.get_result()
        if not result or not result.chain:
            logger.info(f"[Nova-Splitter] 无结果或chain为空")
            return
        
        # 作用范围检查
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = getattr(event, "__is_llm_reply", False)
        
        logger.info(f"[Nova-Splitter] split_scope={split_scope}, is_llm_reply={is_llm_reply}")
        
        if split_scope == "llm_only" and not is_llm_reply:
            logger.info(f"[Nova-Splitter] 非LLM回复，跳过")
            return
        
        # 提取文本内容
        full_text = ""
        for comp in result.chain:
            if isinstance(comp, Plain):
                full_text += comp.text
        
        if not full_text.strip():
            logger.info(f"[Nova-Splitter] 文本为空，跳过")
            return
        
        # 长度检查
        min_length = self.config.get("max_length_no_split", 50)
        logger.info(f"[Nova-Splitter] 文本长度={len(full_text)}, 阈值={min_length}")
        if len(full_text) < min_length:
            logger.info(f"[Nova-Splitter] 文本过短，跳过")
            return
        
        # 标记已处理
        setattr(event, "__nova_splitter_processed", True)
        
        # 选择分段策略
        split_mode = self.config.get("split_mode", "char_count")
        logger.info(f"[Nova-Splitter] 开始处理，模式={split_mode}, 文本长度={len(full_text)}")
        
        try:
            if split_mode == "llm_assist" and self.config.get("enable_llm_assist", False):
                splitter = LLMAssistSplitter(dict(self.config), self.context,
                                              event.unified_msg_origin, self.config)
                split_result = await splitter.split_async(full_text)
            elif split_mode == "punctuation":
                splitter = PunctuationSplitter(dict(self.config))
                split_result = splitter.split(full_text)
            elif split_mode == "hybrid":
                punct_splitter = PunctuationSplitter(dict(self.config))
                split_result = punct_splitter.split(full_text)
                target = self.config.get("target_segments", 3)
                if len(split_result.segments) < target:
                    char_splitter = CharCountSplitter(dict(self.config))
                    split_result = char_splitter.split(full_text)
            else:  # char_count
                splitter = CharCountSplitter(dict(self.config))
                split_result = splitter.split(full_text)
        except Exception as split_err:
            logger.error(f"[Nova-Splitter] 分段策略执行失败: {type(split_err).__name__}: {split_err}")
            logger.error(f"[Nova-Splitter] 分段异常详情:\n{traceback.format_exc()}")
            # 分段失败时不修改原文
            return
        
        segments = split_result.segments
        
        # 如果只有一段，不处理
        if len(segments) <= 1:
            logger.info(f"[Nova-Splitter] 分段结果只有1段，跳过")
            return
        
        # 标点处理
        segments = self.punctuation_processor.process(segments, split_result.split_points)
        
        # 过滤空段
        segments = [s for s in segments if s.strip()]
        
        if len(segments) <= 1:
            logger.info(f"[Nova-Splitter] 标点处理后只剩1段，跳过")
            return
        
        logger.info(f"[Nova-Splitter] 消息被分为 {len(segments)} 段")
        
        # 处理非文本组件
        other_components = [c for c in result.chain if not isinstance(c, Plain)]
        
        # 构建分段消息链
        message_segments = []
        for i, seg_text in enumerate(segments):
            seg_chain = [Plain(seg_text)]
            if i == 0:
                for comp in other_components:
                    if isinstance(comp, Reply):
                        seg_chain.insert(0, comp)
                    else:
                        seg_chain.append(comp)
            message_segments.append(seg_chain)
        
        # 添加回复引用到第一段
        enable_reply = self.config.get("enable_reply", True)
        if enable_reply and message_segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in message_segments[0])
            if not has_reply:
                message_segments[0].insert(0, Reply(id=event.message_obj.message_id))
        
        # 发送前 N-1 段
        for i in range(len(message_segments) - 1):
            seg_chain = message_segments[i]
            seg_text = "".join([c.text for c in seg_chain if isinstance(c, Plain)])
            
            if not seg_text.strip():
                continue
            
            try:
                self._log_segment(i + 1, len(message_segments), seg_chain)
                
                mc = MessageChain()
                mc.chain = seg_chain
                await self.context.send_message(event.unified_msg_origin, mc)
                
                delay = self._calculate_delay(seg_text)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"[Nova-Splitter] 发送分段 {i+1} 失败: {e}")
        
        # 最后一段交给框架
        last_chain = message_segments[-1]
        self._log_segment(len(message_segments), len(message_segments), last_chain, "交给框架")
        
        result.chain.clear()
        result.chain.extend(last_chain)
    
    def _log_segment(self, index: int, total: int, chain: List[BaseMessageComponent], method: str = "主动发送"):
        content = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content += comp.text
            else:
                content += f"[{type(comp).__name__}]"
        
        content = content.replace('\n', '\\n')[:100]
        logger.info(f"[Nova-Splitter] 第 {index}/{total} 段 ({method}): {content}...")
    
    def _calculate_delay(self, text: str) -> float:
        strategy = self.config.get("delay_strategy", "linear")
        
        if strategy == "random":
            return random.uniform(
                self.config.get("random_min", 0.5),
                self.config.get("random_max", 2.0)
            )
        elif strategy == "log":
            base = self.config.get("log_base", 0.3)
            factor = self.config.get("log_factor", 0.5)
            return min(base + factor * math.log(len(text) + 1), 3.0)
        elif strategy == "linear":
            base = self.config.get("linear_base", 0.3)
            factor = self.config.get("linear_factor", 0.05)
            return min(base + len(text) * factor, 3.0)
        else:  # fixed
            return self.config.get("fixed_delay", 1.0)