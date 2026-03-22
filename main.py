"""
Nova Splitter - 智能消息分段 + 引导思考 + 睡眠模式 v1.2.0
作者: Nova for 辉宝主人
功能: 
  1. 按字数均分分段（强制标点边界，绝不在词语中间断开）
  2. 按标点分段
  3. LLM辅助智能分段（带超时保护和完善的异常处理）
  4. 智能标点清理(边界清理/句内保留)
  5. 引导思考（让AI先思考再回复，思维链拦截与缓存）
  6. 沉默机制（AI决定不回复时完全拦截）
  7. 睡眠模式（管理员指令控制AI休眠/唤醒）
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
from astrbot.api.provider import LLMResponse, ProviderRequest, Provider
from astrbot.api.message_components import Plain, BaseMessageComponent, Reply, At


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
        """在理想位置附近找到最佳分割点"""
        start = max(0, ideal_pos - search_range)
        end = min(len(text), ideal_pos + search_range)
        
        best_pos = -1
        best_score = -999
        
        for pos in range(start, end):
            if pos >= len(text):
                continue
            
            char = text[pos]
            score = -999
            
            if char in '\u3002\uff01\uff1f!?\n':
                score = 100
            elif char in '\uff1b;':
                score = 80
            elif char in '\uff0c,\u3001':
                score = 60
            elif char in ' ':
                score = 40
            elif char in '\u2026\u2014\u2013~\uff5e':
                score = 35
            else:
                continue
            
            distance_penalty = abs(pos - ideal_pos) * 2
            score -= distance_penalty
            
            if self.is_inside_pair(text, pos):
                score -= 200
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
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
        
        split_points = []
        for i in range(1, target_segments):
            ideal_pos = i * ideal_length
            best_pos = self.find_best_split_point(text, ideal_pos)
            split_points.append(best_pos)
        
        split_points = sorted(set(split_points))
        split_points = [p for p in split_points 
                       if p >= min_segment_length and p < total_length - min_segment_length]
        
        if not split_points:
            return SplitResult(segments=[text], split_points=[])
        
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
        
        system_prompt = self.config.get("llm_split_system_prompt", "")
        user_template = self.config.get("llm_split_prompt", "\u539f\u6587\uff1a\n{text}")
        llm_timeout = float(self.config.get("llm_timeout", 30.0))
        
        logger.info(f"[Nova-Splitter] \u8bfb\u53d6\u5230\u7684provider_id: '{provider_id}'")
        
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
            logger.info(f"[Nova-Splitter] \u5f00\u59cb\u8c03\u7528LLM\u5206\u6bb5 (provider: {provider_id or 'default'}, timeout: {llm_timeout}s)")
            
            call_kwargs = {
                "prompt": user_prompt,
                "session_id": uuid.uuid4().hex,
            }
            if system_prompt:
                call_kwargs["system_prompt"] = system_prompt
            
            response = await asyncio.wait_for(
                provider.text_chat(**call_kwargs),
                timeout=llm_timeout
            )
            
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
            
            logger.warning(f"[Nova-Splitter] LLM\u8fd4\u56de\u683c\u5f0f\u5f02\u5e38\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5")
            
        except asyncio.TimeoutError:
            logger.error(f"[Nova-Splitter] LLM\u5206\u6bb5\u8d85\u65f6({llm_timeout}s)\uff0c\u56de\u9000\u5230\u6309\u5b57\u6570\u5206\u6bb5")
        except Exception as e:
            logger.error(f"[Nova-Splitter] LLM\u5206\u6bb5\u5931\u8d25: {type(e).__name__}: {e}")
            logger.debug(f"[Nova-Splitter] LLM\u5206\u6bb5\u5f02\u5e38\u8be6\u60c5:\n{traceback.format_exc()}")
        
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

@register("nova-splitter", "Nova", "智能消息分段 + 引导思考 + 睡眠模式", "1.2.0")
class NovaSplitterPlugin(Star):
    """Nova智能分段插件 v1.2.0"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.punctuation_processor = PunctuationProcessor(dict(config))
        
        # 思维链缓存: {session_id: thought_content}
        self.thought_cache: Dict[str, str] = {}
        
        # 睡眠状态: 全局标志
        self.is_sleeping: bool = False
        
        logger.info("[Nova-Splitter] 插件已初始化 v1.2.0")
        logger.info(f"[Nova-Splitter] 分段模式: {config.get('split_mode', 'char_count')}")
        logger.info(f"[Nova-Splitter] 引导思考: {'启用' if config.get('enable_thought_guide', False) else '关闭'}")
        logger.info(f"[Nova-Splitter] 睡眠模式: {'启用' if config.get('enable_sleep_mode', False) else '关闭'}")
    
    # ==================== 引导思考：注入 system prompt ====================
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """在LLM请求前注入引导思考prompt + 睡眠拦截（含白名单）"""
        
        # 睡眠模式检查
        if self.is_sleeping and self.config.get("enable_sleep_mode", False):
            # 检查白名单放行
            if self._check_sleep_whitelist(event):
                logger.info("[Nova-Splitter] 睡眠模式白名单放行")
            else:
                logger.info("[Nova-Splitter] AI正在睡觉，拦截LLM请求")
                event.stop_event()
                return
        
        # 引导思考：追加到 system_prompt 末尾
        if self.config.get("enable_thought_guide", False):
            guide_prompt = self.config.get("thought_guide_prompt", "")
            if guide_prompt:
                if request.system_prompt:
                    request.system_prompt += "\n\n" + guide_prompt
                else:
                    request.system_prompt = guide_prompt
                logger.info("[Nova-Splitter] 已注入引导思考prompt")
    
    def _check_sleep_whitelist(self, event: AstrMessageEvent) -> bool:
        """检查当前消息是否在睡眠白名单中
        
        白名单群聊：必须@bot才放行
        白名单私聊：直接放行
        """
        # 获取白名单配置
        wl_groups_str = self.config.get("sleep_whitelist_groups", "")
        wl_users_str = self.config.get("sleep_whitelist_users", "")
        wl_groups = [g.strip() for g in wl_groups_str.split(",") if g.strip()]
        wl_users = [u.strip() for u in wl_users_str.split(",") if u.strip()]
        
        group_id = event.get_group_id()
        sender_id = event.get_sender_id()
        
        # 群聊白名单检查
        if group_id and group_id in wl_groups:
            # 群聊必须@bot才放行
            bot_id = event.get_self_id()
            chain = event.get_messages()
            is_at_bot = False
            for comp in chain:
                if isinstance(comp, At) and str(comp.qq) == str(bot_id):
                    is_at_bot = True
                    break
            if is_at_bot:
                logger.info(f"[Nova-Splitter] 白名单群 {group_id} + @bot，放行")
                return True
            else:
                logger.info(f"[Nova-Splitter] 白名单群 {group_id} 但未@bot，不放行")
                return False
        
        # 私聊白名单检查
        if not group_id and sender_id in wl_users:
            logger.info(f"[Nova-Splitter] 白名单用户 {sender_id} 私聊，放行")
            return True
        
        return False
    
    # ==================== 思维链拦截 + 沉默机制 ====================
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """拦截LLM响应：提取思维链、检测沉默、标记LLM回复"""
        setattr(event, "__is_llm_reply", True)
        
        if not self.config.get("enable_thought_guide", False):
            logger.info(f"[Nova-Splitter] 检测到LLM响应，已标记事件")
            return
        
        # 获取回复文本
        completion_text = resp.completion_text
        if not completion_text:
            return
        
        # 提取思维链内容
        tag = self.config.get("thought_tag", "thought")
        thought_pattern = re.compile(
            rf'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>',
            re.DOTALL
        )
        
        thought_match = thought_pattern.search(completion_text)
        thought_content = ""
        reply_content = completion_text
        
        if thought_match:
            thought_content = thought_match.group(1).strip()
            # 从回复中移除思维链部分
            reply_content = thought_pattern.sub('', completion_text).strip()
            
            # 缓存思维链（按 session_id 保留最近1条）
            session_id = event.get_session_id()
            self.thought_cache[session_id] = thought_content
            logger.info(f"[Nova-Splitter] 提取思维链: {thought_content[:80]}...")
            logger.info(f"[Nova-Splitter] 实际回复: {reply_content[:80]}...")
        
        # 检测沉默关键词
        silence_keywords_str = self.config.get("silence_keywords", "沉默,静默,不想回复,不回复,不想搭理,没必要回复")
        silence_keywords = [kw.strip() for kw in silence_keywords_str.split(",") if kw.strip()]
        
        # 检查回复内容是否触发沉默（用 [沉默] 或包含沉默关键词）
        reply_stripped = reply_content.strip()
        should_silence = False
        
        # 检查 [xxx] 格式的沉默标记
        bracket_match = re.match(r'^\[(.+?)\]$', reply_stripped)
        if bracket_match:
            bracket_content = bracket_match.group(1)
            for kw in silence_keywords:
                if kw in bracket_content:
                    should_silence = True
                    break
        
        # 也检查纯文本是否就是沉默关键词
        if not should_silence and reply_stripped:
            for kw in silence_keywords:
                if reply_stripped == kw or reply_stripped == f"[{kw}]":
                    should_silence = True
                    break
        
        if should_silence:
            logger.info(f"[Nova-Splitter] AI决定沉默，拦截回复: {reply_stripped}")
            event.stop_event()
            return
        
        # 将清理后的回复写回
        if thought_match:
            resp.completion_text = reply_content
            logger.info(f"[Nova-Splitter] 已移除思维链，回复内容已更新")
    
    # ==================== 指令：想啥呢 / 在想啥 ====================
    
    @filter.command("想啥呢", alias={"在想啥"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_what_thinking(self, event: AstrMessageEvent):
        """查看AI最近一次的思维链内容（管理员专用）
        
        输出的内容会进入上下文，AI能看到自己的想法被公开了
        """
        setattr(event, "__nova_splitter_own_cmd", True)
        session_id = event.get_session_id()
        thought = self.thought_cache.get(session_id, "")
        
        if thought:
            tag = self.config.get("thought_tag", "thought")
            # 用一种AI能理解的格式输出，让她看到后产生"想法被翻出来了"的羞耻感
            yield event.plain_result(
                f"刚才心里想的是：\n<{tag}>{thought}</{tag}>"
            )
        else:
            yield event.plain_result("最近没有在想什么哦")
    
    # ==================== 指令：睡觉 / 起床 ====================
    
    @filter.command("睡觉", alias={"睡吧", "去睡觉", "晚安"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_sleep(self, event: AstrMessageEvent):
        """让AI进入睡眠模式（管理员专用）"""
        setattr(event, "__nova_splitter_own_cmd", True)
        if not self.config.get("enable_sleep_mode", False):
            yield event.plain_result("睡眠模式未启用")
            return
        
        if self.is_sleeping:
            yield event.plain_result("嗯...已经睡了...")
            return
        
        self.is_sleeping = True
        logger.info("[Nova-Splitter] AI已进入睡眠模式")
        yield event.plain_result("嗯 晚安（")
    
    @filter.command("起床", alias={"醒醒", "早安", "起来"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_wake(self, event: AstrMessageEvent):
        """唤醒AI（管理员专用）"""
        setattr(event, "__nova_splitter_own_cmd", True)
        if not self.config.get("enable_sleep_mode", False):
            yield event.plain_result("睡眠模式未启用")
            return
        
        if not self.is_sleeping:
            yield event.plain_result("本来就醒着呢")
            return
        
        self.is_sleeping = False
        logger.info("[Nova-Splitter] AI已从睡眠模式唤醒")
        yield event.plain_result("唔...早安（")
    
    # ==================== 消息分段处理 ====================
    
    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """处理消息分段 + 完全静默拦截"""
        logger.info(f"[Nova-Splitter] on_decorating_result 触发")
        
        # 完全静默检查：睡眠模式 + sleep_full_silence 开启时
        # 拦截所有非本插件指令的消息发送
        if (self.is_sleeping
            and self.config.get("enable_sleep_mode", False)
            and self.config.get("sleep_full_silence", False)
            and not getattr(event, "__nova_splitter_own_cmd", False)):
            
            # 白名单放行检查
            if not self._check_sleep_whitelist(event):
                logger.info("[Nova-Splitter] 完全静默模式：拦截消息发送")
                result = event.get_result()
                if result and result.chain:
                    result.chain.clear()
                return
        
        try:
            await self._do_split(event)
        except Exception as e:
            logger.error(f"[Nova-Splitter] on_decorating_result 发生未预期异常: {type(e).__name__}: {e}")
            logger.error(f"[Nova-Splitter] 异常详情:\n{traceback.format_exc()}")
    
    async def _do_split(self, event: AstrMessageEvent):
        """实际的分段处理逻辑"""
        if getattr(event, "__nova_splitter_processed", False):
            logger.info(f"[Nova-Splitter] 已处理过，跳过")
            return
        
        result = event.get_result()
        if not result or not result.chain:
            logger.info(f"[Nova-Splitter] 无结果或chain为空")
            return
        
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = getattr(event, "__is_llm_reply", False)
        
        logger.info(f"[Nova-Splitter] split_scope={split_scope}, is_llm_reply={is_llm_reply}")
        
        if split_scope == "llm_only" and not is_llm_reply:
            logger.info(f"[Nova-Splitter] 非LLM回复，跳过")
            return
        
        full_text = ""
        comp_positions = []
        at_text_regex = re.compile(r'\[(?:at|At)[:：]\s*(\d+)\]', re.IGNORECASE)
        
        chain_debug = []
        for comp in result.chain:
            if isinstance(comp, Plain):
                text = comp.text
                last_end = 0
                for m in at_text_regex.finditer(text):
                    before = text[last_end:m.start()]
                    if before:
                        full_text += before
                    at_qq = m.group(1)
                    comp_positions.append((len(full_text), At(qq=at_qq)))
                    logger.info(f"[Nova-Splitter] 解析到文本中的at标记: [at:{at_qq}] 偏移量={len(full_text)}")
                    last_end = m.end()
                remaining = text[last_end:]
                if remaining:
                    full_text += remaining
                chain_debug.append(f"Plain({text[:30]}...)" if len(text) > 30 else f"Plain({text})")
            else:
                chain_debug.append(f"{type(comp).__name__}")
                if not isinstance(comp, Reply):
                    comp_positions.append((len(full_text), comp))
        logger.info(f"[Nova-Splitter] 原始chain结构: {' | '.join(chain_debug)}")
        
        if not full_text.strip():
            logger.info(f"[Nova-Splitter] 文本为空，跳过")
            return
        
        min_length = self.config.get("max_length_no_split", 50)
        logger.info(f"[Nova-Splitter] 文本长度={len(full_text)}, 阈值={min_length}")
        if len(full_text) < min_length:
            logger.info(f"[Nova-Splitter] 文本过短，跳过")
            return
        
        setattr(event, "__nova_splitter_processed", True)
        
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
            else:
                splitter = CharCountSplitter(dict(self.config))
                split_result = splitter.split(full_text)
        except Exception as split_err:
            logger.error(f"[Nova-Splitter] 分段策略执行失败: {type(split_err).__name__}: {split_err}")
            logger.error(f"[Nova-Splitter] 分段异常详情:\n{traceback.format_exc()}")
            return
        
        segments = split_result.segments
        
        if len(segments) <= 1:
            logger.info(f"[Nova-Splitter] 分段结果只有1段，跳过")
            return
        
        segments = self.punctuation_processor.process(segments, split_result.split_points)
        segments = [s for s in segments if s.strip()]
        
        if len(segments) <= 1:
            logger.info(f"[Nova-Splitter] 标点处理后只剩1段，跳过")
            return
        
        logger.info(f"[Nova-Splitter] 消息被分为 {len(segments)} 段")
        
        reply_components = [c for c in result.chain if isinstance(c, Reply)]
        
        seg_ranges = []
        offset = 0
        for seg_text in segments:
            seg_start = full_text.find(seg_text, offset)
            if seg_start == -1:
                seg_start = offset
            seg_end = seg_start + len(seg_text)
            seg_ranges.append((seg_start, seg_end))
            offset = seg_end
        
        logger.info(f"[Nova-Splitter] 分段范围: {seg_ranges}")
        logger.info(f"[Nova-Splitter] 非文本组件位置: {[(pos, type(c).__name__) for pos, c in comp_positions]}")
        
        message_segments = []
        for i, seg_text in enumerate(segments):
            seg_chain = []
            seg_start, seg_end = seg_ranges[i]
            
            for comp_offset, comp in comp_positions:
                if isinstance(comp, Reply):
                    continue
                if seg_start <= comp_offset < seg_end:
                    text_before = seg_text[:comp_offset - seg_start]
                    text_after = seg_text[comp_offset - seg_start:]
                    if text_before:
                        seg_chain.append(Plain(text_before))
                    seg_chain.append(comp)
                    seg_text = text_after
                    seg_start = comp_offset
            
            if seg_text:
                seg_chain.append(Plain(seg_text))
            
            if not seg_chain:
                seg_chain.append(Plain(""))
            
            message_segments.append(seg_chain)
        
        failed_texts = []
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
                failed_texts.append(seg_text)
        
        last_chain = message_segments[-1]
        
        if failed_texts:
            last_text = "".join([c.text for c in last_chain if isinstance(c, Plain)])
            combined_text = "\n".join(failed_texts) + "\n" + last_text
            last_chain = [Plain(combined_text)]
            logger.info(f"[Nova-Splitter] 有 {len(failed_texts)} 段发送失败，已合并到最后一段")
        
        for reply_comp in reply_components:
            last_chain.insert(0, reply_comp)
        
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
        else:
            return self.config.get("fixed_delay", 1.0)