"""
找房智能体核心模块 — 工作流节点 + LLM 客户端 + 智能体入口

本模块实现找房智能体的全部节点业务逻辑和智能体入口 HouseFinderAgent。
并行通过 _run_node() 分发 + ThreadPoolExecutor 实现真并行。

核心流程：

  query_leyoujia ─→ query_community ─┬→ search_and_select_baike ─→ fetch_baike_detail ─→ generate_community_commentary
                                      │
                                      └→ generate_house_comments

并行机制：
  - query_community 后通过 ThreadPoolExecutor 同时运行两分支
  - 房源点评分支内部通过 ThreadPoolExecutor 分批并行调用 LLM
  - TOP5精选为手动触发（/select_top_houses 接口）

数据模型：
  HouseFinderState — 通过 @dataclass 定义全部状态字段

LLM 集成：
  LLMClient — 基于 OpenAI SDK 的统一 LLM 调用封装（兼容 DeepSeek 等），支持计时
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from leyoujia_skill import query_leyoujia, query_leyoujia_community
from baike_tool import BaikeTool
from config import get_llm_api_key, get_llm_api_base_url, get_llm_model_name
from openai import OpenAI
from datetime import datetime


@dataclass
class HouseFinderState:
    """
    找房智能体状态
    """
    community_name: str = ""
    min_price: float = 0
    max_price: float = 10000
    min_area: float = 0
    city: str = "深圳"
    leyoujia_results: List[Dict] = None
    community_info: List[Dict] = None
    community_details: Optional[Dict] = None
    task_id: Optional[str] = None
    community_commentary: str = ""
    house_comments: List[Dict] = None
    top_houses: List[Dict] = None
    baike_lemma_id: Optional[int] = None
    baike_lemma_list: List[Dict] = None
    baike_lemma_content: str = ""
    
    def __post_init__(self):
        if self.leyoujia_results is None:
            self.leyoujia_results = []
        if self.community_info is None:
            self.community_info = []
        if self.house_comments is None:
            self.house_comments = []
        if self.top_houses is None:
            self.top_houses = []
        if self.baike_lemma_list is None:
            self.baike_lemma_list = []

class LLMClient:
    """
    大模型客户端，提供统一的大模型调用入口
    已配置 DeepSeek 大模型，使用 OpenAI SDK
    
    设计特点：
      - 延迟创建 OpenAI 客户端，避免启动时需要配置 API Key
      - 支持动态设置 API 配置
      - 记录每次调用的耗时信息
    """

    def __init__(self, api_base_url: str = None, api_key: str = None):
        """
        初始化大模型客户端（延迟创建，不立即连接）
        
        Args:
            api_base_url: API 基础 URL，若为 None 则从配置模块获取
            api_key: API 密钥，若为 None 则从配置模块获取
        """
        self.api_base_url = api_base_url or get_llm_api_base_url()
        self.api_key = api_key or get_llm_api_key()
        self._client = None  # 延迟创建，避免启动时需要配置
        self.timings: List[Dict] = []  # 记录每次 LLM 调用的耗时信息
        self.default_model = get_llm_model_name()  # 从环境变量读取，在实例化时求值
    
    def _get_client(self) -> OpenAI:
        """
        延迟获取 OpenAI 客户端
        
        只有在实际调用时才创建客户端，避免启动时需要配置 API Key。
        
        Returns:
            OpenAI 客户端实例
        
        Raises:
            OpenAIError: 若 API Key 未配置
        """
        if self._client is None:
            if not self.api_key:
                raise OpenAIError("Missing credentials. Please set LLM_API_KEY environment variable or call set_api_config().")
            if not self.api_base_url:
                raise OpenAIError("Missing API base URL. Please set LLM_API_BASE_URL environment variable or call set_api_config().")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
        return self._client
    
    def set_api_config(self, api_base_url: str, api_key: str):
        """
        设置 API 配置
        
        Args:
            api_base_url: API 基础 URL
            api_key: API 密钥
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self._client = None  # 重置客户端，下次调用时重新创建
    
    def call_llm(self, prompt: str, system_prompt: str, model: str = None, temperature: float = 0.7, node_name: str = "", max_tokens: int = 4096) -> str:
        """
        调用大模型生成响应
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（必需）
            model: 模型名称（可选）
            temperature: 温度参数
            node_name: 调用来源节点名（用于计时关联）
            max_tokens: 最大响应token数
        """
        if not self.api_base_url or not self.api_key:
            return "⚠️ 大模型未配置，请先调用 set_api_config 设置API地址和密钥"

        if model is None:
            model = self.default_model

        start_time = datetime.now()
        try:
            # 记录开始时间
            print(f"⏱️  开始调用大模型 - {model} - {node_name}")

            response = self._get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                timeout=120,
                stream=False
            )

            # 计算并显示耗时
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print(f"✅ 大模型调用完成 - 耗时: {elapsed_time:.2f} 秒")

            # 记录到 timings 列表，供前端展示
            self.timings.append({
                "node_name": node_name or "未知节点",
                "elapsed": round(elapsed_time, 2),
                "timestamp": start_time.strftime("%H:%M:%S")
            })

            return response.choices[0].message.content

        except Exception as e:
            # 计算异常耗时
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print(f"❌ 大模型调用异常 - 耗时: {elapsed_time:.2f} 秒 - 错误: {str(e)}")
            self.timings.append({
                "node_name": node_name or "未知节点",
                "elapsed": round(elapsed_time, 2),
                "timestamp": start_time.strftime("%H:%M:%S"),
                "error": str(e)
            })
            return f"❌ 大模型调用异常: {str(e)}"


def _update_progress(state: HouseFinderState, message: str):
    """推送进度更新到前端（通过task_manager）"""
    if not state.task_id:
        return
    try:
        from task_manager import set_progress
        set_progress(state.task_id, message)
    except Exception:
        pass


# ============================================================
# LangGraph 工作流节点 — 数据获取层
# ============================================================

def query_leyoujia_node(state: HouseFinderState) -> Dict[str, Any]:
    """查询乐有家二手房"""
    results = query_leyoujia(
        community_name=state.community_name,
        city=state.city,
        min_price=state.min_price,
        max_price=state.max_price,
        min_area=state.min_area
    )
    return {"leyoujia_results": results}


def query_community_node(state: HouseFinderState) -> Dict[str, Any]:
    """查询小区信息（包含历史成交）"""
    results = query_leyoujia_community(
        community_name=state.community_name,
        city=state.city
    )
    # 保存基础数据到任务结果，供前端增量展示
    if state.task_id:
        from task_manager import update_task_result
        update_task_result(state.task_id, {
            "community_name": state.community_name,
            "leyoujia_results": state.leyoujia_results,
            "community_info": results,
        })
    return {"community_info": results}


# ============================================================
# 百度百科检索 — Prompt 模板 + 节点
# ============================================================

BAIKE_SELECT_SYSTEM_PROMPT = """你是一位专业的房产信息匹配专家。你需要根据小区信息从百度百科搜索结果中选择最匹配的词条ID。

## 任务说明
根据提供的小区信息（地址、建成年代、开发商等）和百度百科搜索结果，判断哪个词条（lemma_id）最匹配目标小区。

## 输出要求
只输出最匹配的 lemma_id 的整数值，不要输出其他任何内容。"""


BAIKE_SELECT_USER_TEMPLATE = """请从以下百度百科搜索结果中选择最匹配的小区词条ID：

## 目标小区信息
小区名称：{community_name}
所在城市：{city}
小区地址：{address}
建成年代：{build_year}
开发商：{developer}

## 百度百科搜索结果
{lemma_list}

请只输出最匹配的 lemma_id 整数值，不要其他内容。"""


def search_and_select_baike_node(state: HouseFinderState, baike_tool: Optional[BaikeTool] = None, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """
    百度百科检索节点：检索小区相关百科词条并选择最佳匹配
    
    在generate_community_commentary_node之前执行
    """
    _update_progress(state, "正在检索百度百科...")
    
    if not baike_tool:
        baike_tool = BaikeTool()
    
    # 1. 调用百度百科API检索词条列表
    lemma_list = baike_tool.search_lemma_list(
        lemma_title=state.community_name,
        top_k=10
    )
    
    if not lemma_list:
        return {
            "baike_lemma_list": [],
            "baike_lemma_id": None
        }
    
    # 2. 判断结果数量
    if len(lemma_list) == 1:
        # 只有一条结果，直接提取
        lemma_id = lemma_list[0].get("lemma_id")
        return {
            "baike_lemma_list": lemma_list,
            "baike_lemma_id": lemma_id
        }
    else:
        # 多条结果，需要大模型判断
        if not llm_client or not llm_client.api_base_url or not llm_client.api_key:
            print("⚠️ 大模型未配置，直接使用第一条结果")
            lemma_id = lemma_list[0].get("lemma_id")
            return {
                "baike_lemma_list": lemma_list,
                "baike_lemma_id": lemma_id
            }
        
        # 格式化lemma列表供大模型阅读
        lemma_list_text = ""
        for i, lemma in enumerate(lemma_list, 1):
            lemma_list_text += f"{i}. lemma_id: {lemma.get('lemma_id')}\n"
            lemma_list_text += f"   lemma_title: {lemma.get('lemma_title')}\n"
            lemma_list_text += f"   lemma_desc: {lemma.get('lemma_desc')}\n\n"
        
        # 准备小区信息
        comm_info = state.community_info[0] if state.community_info and len(state.community_info) > 0 else {}
        
        user_text = BAIKE_SELECT_USER_TEMPLATE.format(
            community_name=state.community_name,
            city=state.city,
            address=comm_info.get("地址", "未知"),
            build_year=comm_info.get("建成年代", "未知"),
            developer=comm_info.get("开发商", "未知"),
            lemma_list=lemma_list_text
        )
        
        # 调用大模型选择最佳lemma_id
        try:
            response = llm_client.call_llm(
                user_text,
                BAIKE_SELECT_SYSTEM_PROMPT,
                temperature=0.0,
                node_name="百科词条匹配"
            )
            
            # 解析返回的lemma_id
            lemma_id = int(response.strip())
            print(f"✅ 大模型选定 lemma_id: {lemma_id}")
            
        except (ValueError, Exception) as e:
            print(f"⚠️ 大模型选择失败: {e}，使用第一条结果")
            lemma_id = lemma_list[0].get("lemma_id")
        
        return {
            "baike_lemma_list": lemma_list,
            "baike_lemma_id": lemma_id
        }


def fetch_baike_detail_node(state: HouseFinderState, baike_tool: Optional[BaikeTool] = None) -> Dict[str, Any]:
    """
    获取百度百科词条详情节点：根据lemma_id获取词条详细内容
    
    在search_and_select_baike之后执行，在generate_community_commentary之前执行
    
    注意：百度千帆平台目前仅提供搜索接口（get_list_by_title），详情接口尚未开放。
    因此本节点从搜索结果中提取lemma_desc作为词条内容。
    """
    _update_progress(state, "正在获取百科词条详情...")
    
    if not baike_tool:
        baike_tool = BaikeTool()
    
    lemma_id = state.baike_lemma_id
    
    if not lemma_id:
        print("⚠️ lemma_id为空，跳过词条详情获取")
        return {"baike_lemma_content": ""}
    
    # 获取词条详情（由于详情API未开放，将从搜索结果中提取描述）
    content = baike_tool.get_lemma_detail(lemma_id, state.baike_lemma_list)
    
    if content:
        print(f"✅ 成功获取词条详情，长度: {len(content)} 字符")
    else:
        print("⚠️ 获取词条详情失败")
    
    return {"baike_lemma_content": content or ""}


# ============================================================
# LLM 生成层 — 小区点评（Prompt + 节点）
# ============================================================

COMMENTARY_SYSTEM_PROMPT = """
你是一位专业的深圳房产分析师。你将收到一个小区的详细信息，请严格按照模板生成一份专业、简洁的小区点评。

输入信息字段
小区名称

地址

均价

建成年代

开发商

物业公司

绿化率

容积率

在售套数

近12个月成交数据

近12个月成交量

百度百科信息（如果有）

注意事项（必须遵守）
在房源分析和推荐时，需特别关注以下因素：

临街房源：噪音大、灰尘多、价格通常偏低，谨慎推荐

底层房源：潮湿、蚊虫多、安全隐患，价格一般最低

顶楼房源：漏水风险、夏热冬冷，但视野好、无楼上噪音干扰

朝向：南向最佳，北向最差，东西向次之

楼层：中高楼层最受欢迎，兼具采光通风和安全性

装修情况：毛坯价格低但需装修成本，精装修可即住但价格高

年代影响：

2000年前小区：价格偏低但设施老化，需重点检查维护状况

2000-2010年：性价比适中，配套成熟

2010年后：品质较高，价格也较高

输出模板（必须严格遵循）
输出使用 Markdown 格式，按以下结构和换行格式组织内容，总字数控制在 300 字左右。不要增加任何额外的文本或格式。

text
### [小区名称] 点评

**核心指标**  
均价：[xx]元/㎡ | 建成：[年份] | 容积率：[x.x] | 绿化率：[xx]% | 在售：[x]套  

**综合评价**  
[用150字左右概括小区区位、交通、配套、户型等特点，可结合百度百科信息]

**价格走势**  
[结合近12个月成交数据，简要分析价格涨跌趋势和成交量情况]

**优点与不足**  
优点：[列出1-2个核心优势]  
不足：[列出1-2个明显短板，结合注意事项中的特殊房源因素]

**适合人群**  
[1句话明确推荐哪类购房者，如刚需首套、改善置换、投资出租等]
请严格按照以上模板填写，若无相关信息可标注“暂无”，不得改变结构顺序。
"""

COMMENTARY_USER_TEMPLATE = """请为以下小区生成专业点评：

小区名称：{name}
地址：{address}
均价：{avg_price}
建成年代：{build_year}年
开发商：{developer}
物业公司：{property_company}
物业费用：{property_fee}
绿化率：{green_rate}
容积率：{plot_ratio}
附近学校：{nearby_schools}
在售套数：{on_sale_count}套

近12个月成交数据：
{monthly_data}

百度百科信息：
{baike_content}

请生成一份专业的小区点评。"""


# ============================================================
# LLM 生成层 — 房源点评（Prompt + 节点，JSON 输出）
# ============================================================

HOUSE_COMMENTS_SYSTEM_PROMPT = """你是一位专业的深圳房产分析师。你将收到一个小区的详细信息和全部房源列表，请为每套房源生成专业的简短点评。

## 房源信息字段
- 房型
- 面积
- 总价
- 朝向
- 楼层
- 装修情况
- 挂牌时间
- 单价
- 房源URL

## 点评要求
1. 为每套房源给出 50-100 字的简短点评
2. 评级维度分为三档：
   - **优秀**：性价比高、户型方正、楼层适中、无明显瑕疵
   - **良好**：整体还行，无大问题，但有小缺陷（如临街、底层等）
   - **一般**：存在明显问题（临街严重、底层/顶楼、价格偏高、户型差等）
3. 结合小区整体成交数据，判断该房源挂牌价是否合理
4. 指出每套房源的亮点和不足
5. 关注楼层、朝向、装修、临街等关键因素

## 输出格式
按以下 JSON 数组格式输出，每项包含：
- 序号（1, 2, 3...）
- 房型
- 总价
- 评级（优秀/良好/一般）
- 点评内容

不要输出任何其他内容，只输出 JSON 数组。"""


HOUSE_COMMENTS_USER_TEMPLATE = """请为以下小区的全部房源生成专业点评：

小区名称：{name}
均价：{avg_price}
建成年代：{build_year}年
在售套数：{on_sale_count}套

全部房源列表：
{houses_list}

请按要求为每套房源生成点评和评级，输出 JSON 数组格式。"""


def generate_community_commentary_node(state: HouseFinderState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """生成小区点评"""
    _update_progress(state, "正在生成小区点评...")

    community_info = state.community_info
    if not community_info:
        return {"community_commentary": "暂无小区详细信息，无法生成点评。"}

    comm = community_info[0]

    name = comm.get('小区名称', '未知')
    address = comm.get('地址', '未知')  # 修正：使用 '地址' 而非 '小区地址'
    avg_price = comm.get('均价', '未知')  # 修正：使用 '均价' 而非 '上月挂牌均价'
    build_year = comm.get('建成年代', '未知')
    developer = comm.get('开发商', '未知')
    property_company = comm.get('物业公司', '未知')
    green_rate = comm.get('绿化率', '未知')
    plot_ratio = comm.get('容积率', '未知')
    nearby_schools = comm.get('学校', '')
    property_fee = comm.get('物业费用', '')

    # 修正：使用 '近12个月成交数据' 而非 '小区近12个月的每月成交均价'
    monthly_list = comm.get('近12个月成交数据') or []
    monthly_text = "\n".join([f"- {m.get('月份', '')}: 均价{m.get('值', '暂无')}元/㎡" for m in monthly_list]) or "暂无数据"

    user_text = COMMENTARY_USER_TEMPLATE.format(
        name=name,
        address=address or '未知',
        avg_price=avg_price if avg_price else '未知',
        build_year=build_year if build_year else '未知',
        developer=developer if developer else '未知',
        property_company=property_company if property_company else '未知',
        green_rate=green_rate if green_rate else '未知',
        plot_ratio=plot_ratio if plot_ratio else '未知',
        nearby_schools=nearby_schools if nearby_schools else '未知',
        property_fee=property_fee if property_fee else '未知',
        on_sale_count=len(state.leyoujia_results),
        monthly_data=monthly_text,
        baike_content=state.baike_lemma_content if state.baike_lemma_content else '暂无'
    )

    if llm_client and llm_client.api_base_url and llm_client.api_key:
        commentary = llm_client.call_llm(user_text, COMMENTARY_SYSTEM_PROMPT, temperature=0.3, node_name="小区综合点评")
    else:
        commentary = "大模型未配置，无法生成小区点评。"

    # 立即更新任务结果，让前端可以实时展示小区点评
    if state.task_id:
        from task_manager import update_task_result
        update_task_result(state.task_id, {
            "community_commentary": commentary,
            "community_info": community_info,
            "baike_lemma_id": state.baike_lemma_id,
            "baike_lemma_list": state.baike_lemma_list,
            "llm_timings": llm_client.timings.copy() if llm_client else []
        })

    return {"community_commentary": commentary}


def _process_single_batch(batch_idx, start_num, batch, leyoujia_results, name, avg_price, build_year, total_count, total_batches, llm_client, lock, results_list):
    """
    处理单批房源点评（供线程池调用）
    
    此函数作为线程池的工作单元，负责：
    1. 将一批房源数据格式化为 LLM 可理解的文本格式
    2. 调用 LLM 生成该批次房源的点评
    3. 解析 LLM 返回的 JSON 结果
    4. 将结果合并到共享的 results_list（线程安全）
    
    Args:
        batch_idx        : 当前批次索引（从1开始）
        start_num        : 当前批次首条房源的全局序号
        batch            : 当前批次的房源数据列表
        leyoujia_results : 完整的房源原始数据列表（用于补充点评信息）
        name             : 小区名称
        avg_price        : 小区均价
        build_year       : 小区建成年代
        total_count      : 房源总数量
        total_batches    : 总批次数
        llm_client       : LLMClient 实例
        lock             : 线程锁（用于保护 results_list）
        results_list     : 共享结果列表（收集所有批次的点评结果）
    """
    import json
    import re

    houses_list = []
    for i, item in enumerate(batch):
        global_idx = start_num + i
        houses_list.append(
            f"{global_idx}. {item.get('rooms', '')} | {item.get('area', 0)}㎡ | {item.get('price', 0)}万 | "
            f"{item.get('orientation', '')} | {item.get('floor', '')}层 | {item.get('decoration', '未知')} | "
            f"单价{item.get('unit_price', '未知')}元/㎡ | {item.get('url', '')}"
        )
    houses_text = "\n".join(houses_list)

    user_text = HOUSE_COMMENTS_USER_TEMPLATE.format(
        name=name,
        avg_price=avg_price if avg_price else '未知',
        build_year=build_year if build_year else '未知',
        on_sale_count=total_count,
        houses_list=houses_text
    )

    result = llm_client.call_llm(user_text, HOUSE_COMMENTS_SYSTEM_PROMPT, temperature=0.1, node_name=f"房源逐套点评({batch_idx}/{total_batches})", max_tokens=4096)

    batch_comments = []
    try:
        cleaned = result.strip()
        markdown_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if markdown_match:
            cleaned = markdown_match.group(1).strip()
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)
        
        if not cleaned:
            raise ValueError("LLM返回内容中未找到有效JSON数组")
        
        comments = json.loads(cleaned)

        for i, c in enumerate(comments):
            global_idx = start_num + i
            house = leyoujia_results[global_idx - 1] if global_idx <= len(leyoujia_results) else {}
            batch_comments.append({
                'rooms': c.get('房型', ''),
                'price': c.get('总价', ''),
                'rating': c.get('评级', '一般'),
                'comment': c.get('点评', ''),
                'area': house.get('area', ''),
                'orientation': house.get('orientation', ''),
                'floor': house.get('floor', ''),
                'decoration': house.get('decoration', ''),
                'unit_price': house.get('unit_price', ''),
                'url': house.get('url', '')
            })
        print(f"✅ 第{batch_idx}批完成，生成 {len(batch_comments)} 条点评")
    except Exception as e:
        print(f"❌ 第{batch_idx}批房源点评解析失败: {e}")
        print(f"LLM返回内容(前500字): {result[:500]}")

    with lock:
        results_list.append((start_num, batch_comments))


def generate_house_comments_node(state: HouseFinderState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """为每套房源生成专业点评（分批并行处理，每批5套）"""
    _update_progress(state, "正在为房源生成点评...")

    leyoujia_results = state.leyoujia_results
    community_info = state.community_info

    if not leyoujia_results:
        return {"house_comments": []}

    comm = community_info[0] if community_info else {}
    name = comm.get('小区名称', state.community_name)
    avg_price = comm.get('均价', '未知')
    build_year = comm.get('建成年代', '未知')

    batch_size = 5
    all_batches = []
    for i in range(0, len(leyoujia_results), batch_size):
        batch = leyoujia_results[i:i + batch_size]
        start_num = i + 1
        all_batches.append((start_num, batch))

    if len(all_batches) > 1 and len(all_batches[-1][1]) < batch_size:
        last_batch_item = all_batches.pop()  # (start_num, batch)
        last_start, last_batch_list = last_batch_item
        prev_start, prev_batch = all_batches[-1]
        all_batches[-1] = (prev_start, prev_batch + last_batch_list)

    total_batches = len(all_batches)
    print(f"📦 共 {len(leyoujia_results)} 套房源，分 {total_batches} 批并行处理")

    if llm_client and llm_client.api_base_url and llm_client.api_key:
        lock = threading.Lock()
        results_list = []

        with ThreadPoolExecutor(max_workers=total_batches) as executor:
            futures = []
            for batch_idx, (start_num, batch) in enumerate(all_batches, 1):
                future = executor.submit(
                    _process_single_batch,
                    batch_idx, start_num, batch, leyoujia_results, name,
                    avg_price, build_year, len(leyoujia_results), total_batches,
                    llm_client, lock, results_list
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()

        results_list.sort(key=lambda x: x[0])
        all_comments = []
        for _, batch_comments in results_list:
            all_comments.extend(batch_comments)

        print(f"✅ 共生成 {len(all_comments)} 条房源点评")
        # 保存房源点评到任务结果，供前端增量展示
        if state.task_id:
            from task_manager import update_task_result
            update_task_result(state.task_id, {
                "house_comments": all_comments,
                "llm_timings": llm_client.timings.copy(),
            })
        return {"house_comments": all_comments}
    else:
        return {"house_comments": []}


# ============================================================
# LLM 生成层 — TOP5 精选房源（Prompt + 节点，JSON 输出）
# ============================================================

# TOP5精选配置 — 集中管理不同版本的选房策略
TOP_HOUSES_CONFIG = {
    "balance": {
        "name": "综合兼顾性价比",
        "progress_msg": "正在筛选最优房源...",
        "system_prompt": """你是一位专业的深圳房产分析师。你将收到小区信息、全部房源列表以及每套房源的点评，请从中选出最优秀的5套房源。

## 选房标准
1. 优先选择评级为 "优秀" 的房源，其次是 "良好" 的房源
2. 重点均衡兼顾高性价比，同时合理考量户型、楼层、朝向、装修情况
3. 尽量选择不同户型，避免过于集中
4. 严格结合小区均价横向对比，精准判断价格合理性，兼顾居住舒适度与入手成本

## 输出格式
按以下 JSON 数组格式输出，每项包含：
- 序号（1, 2, 3, 4, 5）
- 房型
- 面积
- 总价
- 单价
- 朝向
- 楼层
- 评级
- 链接（引用输入中的房源URL，必须原样复制）
- 房源点评（引用输入中的点评内容，简短概括，50字以内）
- 推荐理由（详细说明为什么选中这套房，结合选房标准分析）

不要输出任何其他内容，只输出 JSON 数组。"""
    },
    "price_sensitive": {
        "name": "价格敏感优先",
        "progress_msg": "正在筛选高性价比房源...",
        "system_prompt": """你是一位专业的深圳房产分析师。你将收到小区信息、全部房源列表以及每套房源的点评，请从中选出最优秀的5套房源。

## 选房标准
1. 优先选择评级为 "优秀" 以及 "良好" 的房源
2. 以价格实惠、低价刚需为核心，优先筛选总价低、单价偏低房源，再兼顾户型、楼层、朝向、装修情况
3. 尽量选择不同户型，避免过于集中
4. 优先挑选低于小区均价、价格优势明显的房源，放宽部分居住细节要求

## 输出格式
按以下 JSON 数组格式输出，每项包含：
- 序号（1, 2, 3, 4, 5）
- 房型
- 面积
- 总价
- 单价
- 朝向
- 楼层
- 评级
- 链接（引用输入中的房源URL，必须原样复制）
- 房源点评（引用输入中的点评内容，简短概括，50字以内）
- 推荐理由（详细说明为什么选中这套房，结合选房标准分析）

不要输出任何其他内容，只输出 JSON 数组。"""
    },
    "quality": {
        "name": "品质居住优先",
        "progress_msg": "正在筛选高品质房源...",
        "system_prompt": """你是一位专业的深圳房产分析师。你将收到小区信息、全部房源列表以及每套房源的点评，请从中选出最优秀的5套房源。

## 选房标准
1. 你可以不用特别在意评级，主要看居住品质
2. 弱化价格高低影响，优先看重优质户型、好楼层、南北通透好朝向、精装优质房源，优先保障居住品质
3. 尽量选择不同户型，避免过于集中
4. 无需刻意压低价格，优先挑选房源硬件条件优质、居住体验佳的房源，价格合理即可

## 输出格式
按以下 JSON 数组格式输出，每项包含：
- 序号（1, 2, 3, 4, 5）
- 房型
- 面积
- 总价
- 单价
- 朝向
- 楼层
- 评级
- 链接（引用输入中的房源URL，必须原样复制）
- 房源点评（引用输入中的点评内容，简短概括，50字以内）
- 推荐理由（详细说明为什么选中这套房，结合选房标准分析）

不要输出任何其他内容，只输出 JSON 数组。"""
    }
}

TOP_HOUSES_USER_TEMPLATE = """请从以下房源中选出最优秀的5套：

小区名称：{name}
均价：{avg_price}

全部房源点评（已评级）：
{houses_with_comments}

请按选房标准选出最优秀的5套，并给出推荐理由。"""


def select_top_houses_node(
    state: HouseFinderState, 
    llm_client: Optional[LLMClient] = None,
    version: str = "balance"
) -> Dict[str, Any]:
    """
    通用 TOP5 房源选择节点
    
    根据指定版本选择对应的选房策略，调用核心实现获取TOP5精选房源。
    
    Args:
        state     : HouseFinderState 状态对象
        llm_client: LLMClient 实例（可选）
        version   : 选房版本，可选值: "balance"（综合兼顾）、"price_sensitive"（价格敏感）、"quality"（品质优先）
    
    Returns:
        Dict 包含 top_houses 和 selection_version
    """
    config = TOP_HOUSES_CONFIG.get(version, TOP_HOUSES_CONFIG["balance"])
    _update_progress(state, config["progress_msg"])
    return _select_top_houses_impl(state, llm_client, config["name"], config["system_prompt"])


# 兼容旧接口的包装函数（保持向后兼容）
def select_top_houses_balance_node(state: HouseFinderState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """从已点评的房源中选出最优秀的5套（综合兼顾性价比版本）"""
    return select_top_houses_node(state, llm_client, "balance")


def select_top_houses_price_sensitive_node(state: HouseFinderState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """从已点评的房源中选出最优秀的5套（价格敏感优先版本）"""
    return select_top_houses_node(state, llm_client, "price_sensitive")


def select_top_houses_quality_node(state: HouseFinderState, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """从已点评的房源中选出最优秀的5套（品质居住版本）"""
    return select_top_houses_node(state, llm_client, "quality")


def _select_top_houses_impl(
    state: HouseFinderState, 
    llm_client: Optional[LLMClient] = None, 
    version: str = "综合兼顾性价比",
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    TOP5房源选择核心实现
    
    Args:
        state        : HouseFinderState 状态对象
        llm_client   : LLMClient 实例（可选）
        version      : 选房版本名称（用于日志和进度显示）
        system_prompt: 系统提示词（可选，若未提供则使用默认的综合兼顾性价比提示词）
    
    Returns:
        Dict 包含 top_houses 和 selection_version
    """
    house_comments = state.house_comments
    leyoujia_results = state.leyoujia_results
    community_info = state.community_info

    # 检查必要数据是否存在
    if not house_comments or len(house_comments) == 0:
        print(f"❌ TOP5精选失败: house_comments 为空")
        return {"top_houses": [], "selection_version": version}
    
    if not leyoujia_results or len(leyoujia_results) == 0:
        print(f"❌ TOP5精选失败: leyoujia_results 为空")
        return {"top_houses": [], "selection_version": version}
    
    print(f"✅ TOP5精选开始: {len(house_comments)} 套房源点评, {len(leyoujia_results)} 套房源数据")

    comm = community_info[0] if community_info else {}
    name = comm.get('小区名称', state.community_name)
    avg_price = comm.get('均价', '未知')

    houses_with_comments = []
    for i, (comment, house) in enumerate(zip(house_comments, leyoujia_results), 1):
        houses_with_comments.append(
            f"{i}. 房型:{comment.get('rooms', '')} | 面积:{house.get('area', 0)}㎡ | 总价:{comment.get('price', '')} | "
            f"单价:{house.get('unit_price', '未知')}元/㎡ | 朝向:{house.get('orientation', '')} | 楼层:{house.get('floor', '')}层 | "
            f"评级:{comment.get('rating', '一般')} | 点评:{comment.get('comment', '')} | 链接:{house.get('url', '')}"
        )
    houses_text = "\n".join(houses_with_comments)

    user_text = TOP_HOUSES_USER_TEMPLATE.format(
        name=name,
        avg_price=avg_price if avg_price else '未知',
        houses_with_comments=houses_text
    )

    # 使用传入的 system_prompt，若未提供则使用默认的综合兼顾性价比提示词
    if system_prompt is None:
        system_prompt = TOP_HOUSES_CONFIG["balance"]["system_prompt"]

    if llm_client and llm_client.api_base_url and llm_client.api_key:
        _update_progress(state, f"正在调用大模型筛选最优房源...({version})")
        result = llm_client.call_llm(user_text, system_prompt, temperature=0.1, node_name=f"TOP5精选房源({version})", max_tokens=2048)
        try:
            import json
            import re
            cleaned = result.strip()
            markdown_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
            if markdown_match:
                cleaned = markdown_match.group(1).strip()
            array_match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned, re.DOTALL)
            if array_match:
                cleaned = array_match.group(0)
            top_houses = json.loads(cleaned)
            for top_house in top_houses:
                rooms = top_house.get('房型', '')
                price = top_house.get('总价', '')
                for i, (comment, house) in enumerate(zip(house_comments, leyoujia_results)):
                    if comment.get('rooms', '') == rooms and comment.get('price', '') == price:
                        if not top_house.get('房源点评'):
                            top_house['房源点评'] = comment.get('comment', '')
                        if not top_house.get('链接'):
                            top_house['链接'] = house.get('url', '')
                        if not top_house.get('装修'):
                            top_house['装修'] = house.get('decoration', '')
                        break
            return {"top_houses": top_houses, "selection_version": version}
        except Exception as e:
            print(f"❌ TOP房源解析失败: {e}")
            print(f"LLM返回内容(前500字): {result[:500]}")
            return {"top_houses": [], "selection_version": version}
    else:
        return {"top_houses": [], "selection_version": version}


# ============================================================
# 智能体入口 — 组装工作流并暴露 search() 接口
# ============================================================

class HouseFinderAgent:
    """
    找房智能体
    整合乐有家二手房平台信息与百度百科数据，通过大模型为用户提供购房决策建议
    节点编排通过 _run_node() 分发 + ThreadPoolExecutor 实现真并行
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.baike_tool = BaikeTool()
    
    def search(self, community_name: str, min_price: float = 0, max_price: float = 10000, min_area: float = 0, city: str = "深圳", community_details: Optional[Dict] = None, task_id: Optional[str] = None) -> Dict:
        initial_state = HouseFinderState(
            community_name=community_name,
            min_price=min_price,
            max_price=max_price,
            min_area=min_area,
            city=city,
            community_details=community_details,
            task_id=task_id,
        )
        
        # 串行前序（数据依赖）
        print("🔄 执行前序节点...")
        state = self._run_node("query_leyoujia", initial_state)
        state = self._run_node("query_community", state)
        
        if task_id:
            self._update_task_progress(task_id, state)
        
        # ThreadPoolExecutor 真并行（LangGraph Send 在图内只是扇出，不是多线程）
        print("🔄 并行执行小区点评和房源点评分支...")
        
        def run_community_branch(s):
            s = self._run_node("search_and_select_baike", s)
            s = self._run_node("fetch_baike_detail", s)
            return self._run_node("generate_community_commentary", s)
        
        def run_house_branch(s):
            return self._run_node("generate_house_comments", s)
        
        community_result = None
        house_result = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_community = executor.submit(run_community_branch, state)
            future_house = executor.submit(run_house_branch, state)
            
            for future in as_completed([future_community, future_house]):
                try:
                    result = future.result()
                    if result.community_commentary:
                        community_result = result
                        print("✅ 小区点评分支完成")
                        if task_id:
                            self._update_task_progress(task_id, result)
                    if result.house_comments:
                        house_result = result
                        print("✅ 房源点评分支完成")
                        if task_id:
                            self._update_task_progress(task_id, result)
                except Exception as e:
                    print(f"❌ 分支执行失败: {e}")
        
        final_state = state
        if community_result:
            final_state.community_commentary = community_result.community_commentary
            final_state.baike_lemma_id = community_result.baike_lemma_id
            final_state.baike_lemma_list = community_result.baike_lemma_list
            final_state.baike_lemma_content = community_result.baike_lemma_content
        
        if house_result:
            final_state.house_comments = house_result.house_comments
        
        llm_timings = self.llm_client.timings.copy()
        self.llm_client.timings.clear()
        
        return {
            "community_commentary": final_state.community_commentary,
            "house_comments": final_state.house_comments,
            "community_info": final_state.community_info,
            "leyoujia_results": final_state.leyoujia_results,
            "community_name": final_state.community_name,
            "baike_lemma_id": final_state.baike_lemma_id,
            "baike_lemma_list": final_state.baike_lemma_list,
            "llm_timings": llm_timings,
        }
    
    def _run_node(self, node_name: str, state: HouseFinderState) -> HouseFinderState:
        """执行单个节点并应用状态更新"""
        nodes = {
            "query_leyoujia": query_leyoujia_node,
            "query_community": query_community_node,
            "search_and_select_baike": lambda s: search_and_select_baike_node(s, self.baike_tool, self.llm_client),
            "fetch_baike_detail": lambda s: fetch_baike_detail_node(s, self.baike_tool),
            "generate_community_commentary": lambda s: generate_community_commentary_node(s, self.llm_client),
            "generate_house_comments": lambda s: generate_house_comments_node(s, self.llm_client),
        }
        node_func = nodes[node_name]
        print(f"  → 执行节点: {node_name}")
        updates = node_func(state)
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
    
    def _update_task_progress(self, task_id: str, state):
        """
        实时更新任务进度和部分结果，用于增量展示
        
        Args:
            task_id: 任务ID
            state: 当前状态对象（可能是 HouseFinderState 或 dict）
        """
        from task_manager import update_task_result
        
        partial_result = {}
        
        def get_attr(key, default=None):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)
        
        community_name = get_attr("community_name")
        if community_name:
            partial_result["community_name"] = community_name
        
        community_commentary = get_attr("community_commentary")
        if community_commentary:
            partial_result["community_commentary"] = community_commentary
        
        community_info = get_attr("community_info")
        if community_info:
            partial_result["community_info"] = community_info
        
        baike_lemma_id = get_attr("baike_lemma_id")
        if baike_lemma_id:
            partial_result["baike_lemma_id"] = baike_lemma_id
        
        baike_lemma_list = get_attr("baike_lemma_list")
        if baike_lemma_list:
            partial_result["baike_lemma_list"] = baike_lemma_list
        
        house_comments = get_attr("house_comments")
        if house_comments:
            partial_result["house_comments"] = house_comments
        
        leyoujia_results = get_attr("leyoujia_results")
        if leyoujia_results:
            partial_result["leyoujia_results"] = leyoujia_results
        
        partial_result["llm_timings"] = self.llm_client.timings.copy()
        
        if partial_result:
            update_task_result(task_id, partial_result)

    def select_top_houses_balance(self, state: HouseFinderState) -> Dict:
        """综合兼顾性价比版本 - 选出最优5套房源"""
        result = select_top_houses_balance_node(state, self.llm_client)
        timings = self.llm_client.timings[-1:] if self.llm_client.timings else []
        return {**result, "llm_timings": timings}

    def select_top_houses_price_sensitive(self, state: HouseFinderState) -> Dict:
        """价格敏感优先版本 - 侧重低价、划算、省钱"""
        result = select_top_houses_price_sensitive_node(state, self.llm_client)
        timings = self.llm_client.timings[-1:] if self.llm_client.timings else []
        return {**result, "llm_timings": timings}

    def select_top_houses_quality(self, state: HouseFinderState) -> Dict:
        """品质居住版本 - 侧重居住、户型、楼层、装修"""
        result = select_top_houses_quality_node(state, self.llm_client)
        timings = self.llm_client.timings[-1:] if self.llm_client.timings else []
        return {**result, "llm_timings": timings}
