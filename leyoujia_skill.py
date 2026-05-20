"""
乐有家 SKILL 模块 — 封装乐有家二手房平台 API

本模块提供乐有家开放平台的两大核心查询能力：
  1. query_leyoujia           — 搜索小区房源列表（支持价格/面积/户型筛选，自动分页）
  2. query_leyoujia_community — 搜索小区详情（含近12个月成交/挂牌/供需月度数据）

额外工具函数：
  - format_community_info()     — 将原始API字段映射为统一的中文字段名
  - fallback_community_search() — 当小区查询接口不可用时，通过房源搜索接口兜底
  - _parse_results()           — 解析房源搜索返回的原始列表为标准化字典列表

API 认证方式：使用 X-Api-Key 请求头鉴权。
数据流：
  API原始响应 → 字段映射/格式化 → 标准化字典列表 → 传入工作流节点函数
"""

import json
import os
from datetime import datetime
from typing import List, Dict
from config import get_leyoujia_api_key
from utils.http_client import HttpClient

# ——————————————————————————————————————————————
# API 基础配置
# ——————————————————————————————————————————————
BASE_URL = "https://wap.leyoujia.com/wap/openclaw/ai"
SUPPORTED_CITIES = ["深圳", "中山", "东莞", "惠州", "广州", "佛山", "清远", "珠海", "江门", "长沙", "南京"]

# ——————————————————————————————————————————————
# 调试配置（开发时可开启 DEBUG_MODE=True 保存API原始响应）
# ——————————————————————————————————————————————
DEBUG_MODE = False
DEBUG_DIR = "skill_debug_logs"


# 全局 HTTP 客户端实例（懒加载）
_http_client_instance = None

def _get_http_client() -> HttpClient:
    """
    获取乐有家 API 的 HTTP 客户端实例（单例模式）
    
    Returns:
        HttpClient 实例
        
    Raises:
        ValueError: 若 API Key 未配置
    """
    global _http_client_instance
    
    if _http_client_instance is None:
        api_key = get_leyoujia_api_key()
        
        # 验证 API Key 是否配置
        if not api_key:
            print("⚠️ 警告: LEYOUJIA_API_KEY 环境变量未设置，API 调用可能失败")
        
        headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        }
        _http_client_instance = HttpClient(base_url=BASE_URL, default_headers=headers, timeout=10, debug=DEBUG_MODE)
    
    return _http_client_instance


def _save_debug_data(api_name: str, params: Dict, response_data: Dict):
    """
    将 API 返回的原始数据保存为 JSON 文件，用于离线调试

    Args:
        api_name      : API 名称标识（如 "house_search"、"communitySearch"）
        params        : 发送的请求参数
        response_data : API 返回的响应 JSON
    """
    if not DEBUG_MODE:
        return

    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{DEBUG_DIR}/{api_name}_{timestamp}.json"

        debug_data = {
            "timestamp": timestamp,
            "api": api_name,
            "params": params,
            "response": response_data
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)

        print(f"调试数据已保存: {filename}")
    except Exception as e:
        print(f"保存调试数据失败: {str(e)}")


# ============================================================
# 核心对外接口
# ============================================================

def query_leyoujia(community_name: str, city: str = "深圳", **kwargs) -> List[Dict]:
    """
    查询乐有家平台的二手房房源列表（自动分页获取）

    调用 /house/search 接口，根据小区名称和筛选条件返回房源列表。
    目前 API 分页不生效（多页为重复数据），因此只获取第1页。

    Args:
        community_name: 小区名称
        city          : 城市名称（必须在 SUPPORTED_CITIES 列表内）
        **kwargs      : 可选筛选条件
            min_price  — 最低价格(万)
            max_price  — 最高价格(万)
            min_area   — 最小面积(?)
            room       — 户型筛选（如 "3"）

    Returns:
        标准化房源列表，每项包含 platform、community、price、area、
        rooms、orientation、decoration、url 等字段
    """
    if city not in SUPPORTED_CITIES:
        print(f"警告：城市 {city} 不在支持列表中，使用默认城市深圳")
        city = "深圳"

    min_price = kwargs.get('min_price', 0)
    max_price = kwargs.get('max_price', 10000)
    min_area = kwargs.get('min_area', 0)
    room = kwargs.get('room', "")

    http_client = _get_http_client()
    all_results = []
    page = 1
    page_size = 30  # API 实际返回上限
    max_pages = 1   # 只获取1页（API分页暂不生效，多页为重复数据）

    while page <= max_pages:
        data = {
            "type": "esf",
            "city": city,
            "keyword": community_name,
            "priceMin": min_price,
            "priceMax": max_price,
            "areaMin": min_area,
            "page": page,
            "pageSize": page_size
        }

        if room:
            data["room"] = str(room)

        success, result = http_client.post("house/search", data=data)
        
        if success:
            if page == 1:
                _save_debug_data("house_search", data, result)

            items = result.get("list", [])
            if items:
                parsed = _parse_results({"list": items}, community_name)
                all_results.extend(parsed)

                if len(items) < page_size:
                    break
                page += 1
            else:
                break
        else:
            print(f"乐有家API调用失败: {result.get('error', '未知错误')}")
            break

    print(f"乐有家API共获取 {len(all_results)} 条房源数据")
    return all_results


def query_leyoujia_community(community_name: str, city: str = "深圳") -> List[Dict]:
    """
    查询乐有家平台的小区详细信息

    调用 /communitySearch 接口，返回小区的档案、价格、市场数据等。
    当小区查询接口不可用时，自动降级为 fallback_community_search()。

    Args:
        community_name: 小区名称
        city          : 城市名称

    Returns:
        格式化后的小区信息列表（最多返回全部匹配结果），每项包含：
          - 基础档案：小区名称、地址、建成年代、开发商、物业公司等
          - 价格相关：均价、近12个月成交/挂牌/供需月度数据
          - 周边配套：学校（地铁/商圈暂未返回）
    """
    http_client = _get_http_client()
    
    data = {
        "city": city,
        "communityKeyword": community_name
    }

    success, result = http_client.post("communitySearch", data=data)

    if not success:
        error_msg = result.get('error', '未知错误')
        print(f"乐有家小区查询API调用失败: {error_msg}")
        return []

    _save_debug_data("communitySearch", data, result)

    data_list = result.get("list", [])
    if not data_list:
        print(f"乐有家小区查询API返回空数据")
        return []

    formatted_list = []
    for item in data_list:
        formatted = format_community_info(item)
        formatted_list.append(formatted)
    return formatted_list


# ============================================================
# 数据格式化
# ============================================================

def format_community_info(raw_item: Dict) -> Dict:
    """
    将乐有家 API 原始字段映射为统一的中文字段名

    原始 API 返回的字段名较长且携带业务前缀（如 "小区近12个月的每月二手房成交量"），
    本函数将其映射为短中文名（如 "近12个月成交量"），同时处理时间戳转换和数字格式化。

    Args:
        raw_item: 乐有家 API 返回的原始小区字典

    Returns:
        格式化后的小区信息字典，按业务含义分为5组：
          1. 小区名称、所在区域
          2. 基础档案（地址、建成年代、物业、开发商）
          3. 价格相关（均价、近12个月各类月度数据）
          4. 周边配套（学校、地铁、商圈）
          5. 其他信息（绿化率、容积率、停车位等）
    """
    # 处理建筑年代 — API返回Unix毫秒时间戳，转换为年份字符串
    build_year = raw_item.get("建筑年代", "")
    if isinstance(build_year, (int, float)) and build_year > 0:
        try:
            build_year = str(datetime.fromtimestamp(build_year / 1000).year)
        except:
            build_year = ""

    # 处理均价 — 数字格式化为 "xxxxx元/㎡"
    avg_price = raw_item.get("上月挂牌均价", "")
    if isinstance(avg_price, (int, float)):
        avg_price = f"{round(avg_price)}元/㎡"

    # 处理绿化率 — 数字格式化为 "xx%"
    green_rate = raw_item.get("绿化率", "")
    if isinstance(green_rate, (int, float)):
        green_rate = f"{green_rate}%"

    return {
        # 1. 小区名称、所在区域
        "小区名称": raw_item.get("小区名称", ""),
        "所在区域": raw_item.get("所在区域", ""),

        # 2. 基础档案
        "地址": raw_item.get("小区地址", ""),
        "建成年代": build_year,
        "物业公司": raw_item.get("物业公司", ""),
        "开发商": raw_item.get("开发商户", ""),

        # 3. 价格相关（字段映射：长API名→短中文名）
        "均价": avg_price,
        "在售数量": raw_item.get("在售数量", ""),
        "在租数量": raw_item.get("在租数量", ""),
        "近12个月成交数据": raw_item.get("小区近12个月的每月成交均价", []),
        "近12个月挂牌价": raw_item.get("小区近12个月的每月挂牌价", []),
        "近12个月成交量": raw_item.get("小区近12个月的每月二手房成交量", []),
        "近12个月二手房挂牌量": raw_item.get("小区近12个月的每月二手房挂牌量", []),
        "近12个月新增二手房挂牌量": raw_item.get("小区近12个月的每月新增二手房挂牌量", []),

        # 4. 周边配套
        "地铁": "",  # 当前API未返回地铁信息
        "学校": ", ".join(raw_item.get("附近学校", [])),  # 数组→逗号分隔字符串
        "商圈": "",  # 当前API未返回商圈信息

        # 5. 其他信息
        "绿化率": green_rate,
        "容积率": raw_item.get("容积率", ""),
        "停车位": raw_item.get("停车位", ""),
        "楼栋数": raw_item.get("楼栋数", ""),
        "物业费用": raw_item.get("物业费用", ""),
        "小区ID": raw_item.get("id", ""),
        "小区详情地址": raw_item.get("小区详情地址", ""),
        "小区图片": raw_item.get("小区图片", ""),
    }


# ============================================================
# 降级兜底
# ============================================================

def fallback_community_search(community_name: str, city: str) -> List[Dict]:
    """
    小区查询接口降级兜底方案

    当 /communitySearch 接口不可用时，通过 /house/search 接口获取房源数据，
    然后从首套房源中提取小区信息作为降级结果。

    Args:
        community_name: 小区名称
        city          : 城市

    Returns:
        小区信息列表（仅含从房源数据中能提取到的字段）
    """
    print(f"使用降级方案查询小区: {community_name}")

    http_client = _get_http_client()
    
    data = {
        "type": "esf",
        "city": city,
        "keyword": community_name,
        "priceMin": 0,
        "priceMax": 10000,
        "page": 1,
        "pageSize": 10
    }

    success, result = http_client.post("house/search", data=data)

    if success:
        items = result.get("list", [])

        if items:
            community_info = {}
            first_item = items[0]

            community_info["小区名称"] = first_item.get("小区名称", community_name)
            community_info["所在区域"] = first_item.get("区域", "")
            community_info["地址"] = first_item.get("地址", "")
            community_info["建成年代"] = first_item.get("竣工日期", "")
            community_info["均价"] = first_item.get("单价", "")
            community_info["在售数量"] = len(items)

            return [community_info]
        else:
            print(f"降级方案也未找到匹配房源: {community_name}")
            # 返回包含小区名称的占位信息，允许用户继续搜索流程
            return [{
                "小区名称": community_name,
                "所在区域": "",
                "地址": "",
                "建成年代": "",
                "均价": "未知",
                "在售数量": 0,
                "_fallback": True,  # 标记为降级方案返回的占位数据
                "_message": "小区信息暂不可用，仍可尝试搜索房源"
            }]
    else:
        error_msg = result.get('error', '未知错误')
        print(f"降级查询失败: {error_msg}")
        # 返回包含小区名称的占位信息，允许用户继续搜索流程
        return [{
            "小区名称": community_name,
            "所在区域": "",
            "地址": "",
            "建成年代": "",
            "均价": "未知",
            "在售数量": 0,
            "_fallback": True,  # 标记为降级方案返回的占位数据
            "_message": f"查询异常: {error_msg}"
        }]


# ============================================================
# 内部解析
# ============================================================

def _parse_results(result: Dict, community_name: str) -> List[Dict]:
    """
    解析乐有家房源搜索 API 返回的原始列表，输出标准化字典列表

    从API字段（如 "室"、"卫"、"总价"、"建筑面积"、"朝向" 等）映射到统一的英文字段名。
    同时处理：
      - 户型字段拼接（室+厅）
      - 竣工年份提取（从时间戳取前4位）
      - 小区名称清理（去除搜索高亮HTML标签）
      - 无效数据过滤（价格或面积为0的房源）

    Args:
        result         : API 返回的原始数据（{"list": [...]}）
        community_name : 小区名称（用于兜底，当API返回的小区名称为空时使用）

    Returns:
        标准化房源列表，每项包含：
          - platform    : 数据来源平台标识（固定为 "乐有家"）
          - community   : 小区名称（已清理HTML标签）
          - price       : 总价（万元，浮点数）
          - area        : 建筑面积（㎡，浮点数）
          - rooms       : 户型（如 "3室2厅"）
          - floor       : 楼层（当前API未返回，留空）
          - orientation : 朝向（如 "南向"、"南北通透"）
          - year        : 竣工年份（整数，0表示未知）
          - decoration  : 装修情况（如 "精装"、"毛坯"）
          - description : 房源描述（优先取房源亮点，其次取标题）
          - url         : 详情页链接
    """
    raw_list = result.get("list", [])
    parsed = []

    for item in raw_list:
        try:
            # 户型 — 拼接 "室" + "卫" 字段
            rooms = f"{item.get('室', '')}室{item.get('卫', '')}厅" if item.get('室') else ''

            # 竣工年份 — 从 "竣工日期" 时间戳提取年份（取前4位）
            year = 0
            try:
                timestamp = item.get('竣工日期', 0)
                if timestamp:
                    year = int(str(timestamp)[:4])
            except:
                pass

            # 小区名称 — 去除搜索高亮标记 <font color="red">
            community = item.get('小区名称', community_name)
            if isinstance(community, str):
                community = community.replace('<font color="red">', '').replace('</font>', '')

            parsed_item = {
                'platform': '乐有家',
                'community': community,
                'price': float(item.get('总价', 0)),
                'area': float(item.get('建筑面积', 0)),
                'rooms': rooms,
                'floor': '',  # 当前API未返回楼层信息
                'orientation': item.get('朝向', ''),
                'year': year,
                'decoration': item.get('装修', ''),
                'description': item.get('房源亮点', item.get('标题', '')),
                'url': item.get('详情地址', '')
            }

            # 过滤无效数据 — 价格或面积为0的视为无效
            if parsed_item['price'] > 0 and parsed_item['area'] > 0:
                parsed.append(parsed_item)
        except Exception as e:
            print(f"解析房源数据失败: {str(e)}")
            continue

    return parsed
