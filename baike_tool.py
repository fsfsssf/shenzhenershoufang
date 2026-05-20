#!/usr/bin/env python3
"""
百度百科工具模块 — 封装百度千帆 AppBuilder 百科 API

本模块提供两个核心能力：
  1. search_lemma_list  — 根据词条名称搜索百科词条列表
  2. get_lemma_detail  — 根据词条ID获取词条详细内容（含内容清洗）

API 认证方式：使用 Authorization: Bearer <API Key> 头部鉴权。
API 文档参考：https://cloud.baidu.com/doc/qianfan-api/s/0mocjdxi9

数据流：
  搜索词条 → 选择匹配ID → 获取详情（含abstract_plain） → 内容清洗 → 传入小区点评节点
"""

import re
from typing import List, Dict, Optional
from config import get_baike_api_key
from utils.http_client import HttpClient


class BaikeTool:
    """
    百度百科工具类

    封装百度千帆 AppBuilder 百科 API 的调用逻辑，包括：
      - 词条列表检索 (get_list_by_title)
      - 词条详细内容获取 (get_content)
      - 内容清洗（去除 HTML 标签、URL、特殊字符、重复段落）
    """

    # API 端点配置
    SEARCH_URL = "https://appbuilder.baidu.com/v2/baike/lemma/get_list_by_title"
    CONTENT_URL = "https://appbuilder.baidu.com/v2/baike/lemma/get_content"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化百度百科工具

        Args:
            api_key: 百度云 API Key，若为 None 则从 config 模块获取
        """
        self.api_key = api_key or get_baike_api_key()
        
        # 检查 API Key 是否配置
        if not self.api_key:
            print("⚠️ 警告: BAIDU_API_KEY 环境变量未设置，百科检索功能将被跳过")
        
        # 初始化 HTTP 客户端
        self._http_client = HttpClient(timeout=10)

    def _get_headers(self) -> Dict[str, str]:
        """获取百度百科 API 请求头"""
        return {"Authorization": f"Bearer {self.api_key}"}

    def search_lemma_list(self, lemma_title: str, top_k: int = 5) -> List[Dict]:
        """
        根据词条名称搜索百科词条列表

        调用百度百科 get_list_by_title 接口，返回与搜索词相关的词条列表。

        Args:
            lemma_title: 词条名称（如小区名）
            top_k      : 返回结果数量，默认5，范围 [1, 100]

        Returns:
            词条列表，每项包含：
              - lemma_id    : 词条ID（用于后续获取详情）
              - lemma_title : 词条标题
              - lemma_desc  : 词条简短描述
              - url         : 百科页面URL

            失败时返回空列表。
        """
        if not self.api_key:
            print("??? BAIDU_API_KEY 未配置，跳过百科检索")
            return []

        params = {"lemma_title": lemma_title, "top_k": top_k}
        headers = self._get_headers()
        
        success, result = self._http_client.get(self.SEARCH_URL, params=params, headers=headers)

        if success:
            # 兼容两种响应格式：直接返回数组 或 嵌套在 code/result 结构中
            if "result" in result:
                items = result.get("result", [])
                return items if isinstance(items, list) else []
            elif result.get("code") in ("0", 0):
                return result.get("result", {}).get("items", [])
            else:
                print(f"? 百度百科API调用失败: {result.get('message', '未知错误')}")
                return []
        else:
            print(f"? 百度百科API请求异常: {result.get('error', '未知错误')}")
            return []

    def get_lemma_detail(self, lemma_id: int, lemma_list: Optional[List[Dict]] = None) -> Optional[str]:
        """
        根据词条 ID 获取百科词条详细内容

        调用百度百科 get_content 接口（search_type=lemmaId），提取 result 中的
        abstract_plain（纯文本摘要，隐藏字段）、abstract、content 等字段，
        然后通过 _clean_content() 清洗后返回。

        Args:
            lemma_id  : 词条ID
            lemma_list: 搜索结果列表，作为 API 失败时的降级数据源

        Returns:
            清洗后的词条纯文本内容；失败时返回 None

        容错策略：
          1. 优先通过 API 获取详细 content
          2. 若 API 失败，回退到搜索结果中的 lemma_desc 字段
        """
        if not self.api_key or not lemma_id:
            return None

        params = {"search_type": "lemmaId", "search_key": lemma_id}
        headers = self._get_headers()
        
        success, result = self._http_client.get(self.CONTENT_URL, params=params, headers=headers)

        if success:
            if "result" in result:
                content = result.get("result", "")
                if content and isinstance(content, dict):
                    text_parts = []

                    # abstract_plain 是纯文本摘要（隐藏字段），字段优先级最高
                    if "abstract_plain" in content:
                        text_parts.append(content["abstract_plain"])
                    if "abstract" in content:
                        text_parts.append(content["abstract"])
                    if "content" in content:
                        text_parts.append(content["content"])

                    cleaned_content = self._clean_content("\n".join(text_parts))
                    print(f"? 获取词条详情成功，长度: {len(cleaned_content)} 字符")
                    return cleaned_content
        else:
            print(f"? 百度百科API请求异常: {result.get('error', '未知错误')}")

        # 降级方案：从搜索结果列表中匹配 lemma_id 并提取 lemma_desc
        if lemma_list:
            for lemma in lemma_list:
                if lemma.get("lemma_id") == lemma_id:
                    desc = lemma.get("lemma_desc", "")
                    if desc:
                        return self._clean_content(desc)

        return None

    def _clean_content(self, content: str) -> str:
        """
        清洗百科内容中的无意义字符

        清洗步骤：
          1. 移除 HTML 标签（如 <p>、<br> 等）
          2. 移除 URL 链接
          3. 压缩多余空白字符（换行、空格）
          4. 移除连续的特殊符号
          5. 去除 BOM 和不可见字符
          6. 去除完全重复的行

        Args:
            content: 原始百科内容

        Returns:
            清洗后的纯文本内容
        """
        if not content:
            return ""

        # 移除 HTML 标签和 URL 链接
        cleaned = re.sub(r'<[^>]*>', '', content)
        cleaned = re.sub(r'https?://[\w\-.~:/?#[\]@!$&\'()*+,;=%]+', '', cleaned)

        # 压缩多余空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[~!@#$%^&*()_+{}|:"<>?\-=\[\]\\;\',./]+', ' ', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)

        # 去除 BOM 和不可见字符
        cleaned = cleaned.replace('\ufeff', '').strip()

        # 去除完全重复的行（保留首次出现）
        lines = cleaned.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())

        return '\n'.join(unique_lines)


if __name__ == "__main__":
    tool = BaikeTool()

    test_lemma = "后海花园"
    print(f"?? 搜索词条: {test_lemma}")
    results = tool.search_lemma_list(test_lemma)

    if results:
        print(f"? 找到 {len(results)} 个词条")
        for i, item in enumerate(results, 1):
            print(f"\n{i}. {item.get('lemma_title')}")
            print(f"   描述: {item.get('lemma_desc')}")
            print(f"   ID: {item.get('lemma_id')}")

            detail = tool.get_lemma_detail(item.get('lemma_id'), results)
            if detail:
                print(f"   详情预览:\n{detail[:100]}...")
    else:
        print("? 未找到词条")
