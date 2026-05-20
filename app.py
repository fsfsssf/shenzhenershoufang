"""
找房智能体 Web 服务 — 基于 Flask 的 REST API (CloudBase 云托管 / Docker)

提供五个核心接口：

  GET  /                   — 渲染前端页面 (index.html)
  POST /search_communities — 搜索小区列表
  POST /search             — 启动异步找房任务
  GET  /task_status/<id>   — 轮询任务状态和结果
  POST /select_top_houses  — TOP5 精选房源

部署方式：CloudBase 云托管 / Docker 容器，gunicorn 监听 80 端口

数据流：
  浏览器 → HTTP 访问服务 → CloudBase 云托管 → gunicorn → Flask (80)
"""

# 优先加载环境变量（从 .env 文件）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载 .env 文件中的环境变量
    print("✅ 已加载 .env 环境变量")
except ImportError:
    print("⚠️ 未安装 python-dotenv，环境变量需要手动设置")

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from leyoujia_ana import HouseFinderAgent
from leyoujia_skill import query_leyoujia_community
from task_manager import create_task, run_task, get_task_status, get_task_result

app = Flask(__name__, template_folder='.')
CORS(app, origins='*')

agent = HouseFinderAgent()


@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')


@app.route('/search_communities', methods=['POST'])
def search_communities():
    """
    搜索小区列表

    接收小区名称和城市，调用乐有家 API 返回匹配的小区列表，
    供前端展示并让用户选择正确的小区。

    请求体 JSON:
        { "community_name": "香蜜新村", "city": "深圳" }

    返回:
        { "success": true, "communities": [{ name, address, avg_price, ... }, ...] }
    """
    try:
        data = request.get_json()
        community_name = data.get('community_name', '').strip()
        city = data.get('city', '深圳')

        if not community_name:
            return jsonify({'error': '请输入小区名称'}), 400

        communities = query_leyoujia_community(community_name, city)

        community_list = []
        for i, community in enumerate(communities, 1):
            community_list.append({
                'index': i,
                'name': community.get('小区名称', ''),
                'address': community.get('地址', ''),
                'avg_price': community.get('均价', ''),
                'build_year': community.get('建成年代', ''),
                'developer': community.get('开发商', ''),
                'property_company': community.get('物业公司', ''),
                'green_rate': community.get('绿化率', ''),
                'plot_ratio': community.get('容积率', ''),
                'schools': community.get('学校', ''),
                'property_fee': community.get('物业费用', ''),
                'full_data': community  # 完整数据供后续工作流使用
            })

        return jsonify({'success': True, 'communities': community_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    启动异步找房任务

    用户在前端确认小区后，创建一个后台线程运行 LangGraph 工作流。
    返回 task_id，前端通过 /task_status/<task_id> 轮询结果。

    请求体 JSON:
        { "community_name": "香蜜新村", "city": "深圳", "community_details": {...} }

    返回:
        { "success": true, "task_id": "uuid-string" }
    """
    try:
        data = request.get_json()
        community_name = data.get('community_name', '').strip()
        city = data.get('city', '深圳')
        community_details = data.get('community_details', None)

        if not community_name:
            return jsonify({'error': '请输入小区名称'}), 400

        task_id = create_task()
        run_task(
            task_id,
            agent.search,
            community_name,
            city=city,
            community_details=community_details,
            task_id=task_id
        )

        return jsonify({'success': True, 'task_id': task_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    """
    轮询任务状态

    前端每2秒调用一次此接口。任务进行中也会返回已生成的部分结果，实现增量展示。

    路径参数:
        task_id: 异步任务ID

    返回:
        进行中: { "status": "running", "progress": "...", "community_commentary": "...", ... }
        已完成: { "status": "done", "community_commentary": "...", "house_comments": [...], ... }
        失败:   { "status": "error", "error": "错误信息" }
    """
    status = get_task_status(task_id)
    if not status:
        return jsonify({'error': '任务不存在'}), 404

    # 获取当前已有的结果（可能是部分结果）
    result = get_task_result(task_id) or {}

    response = {
        'status': status['status'],
        'progress': status.get('progress', ''),
        'community_commentary': result.get('community_commentary', ''),
        'house_comments': result.get('house_comments', []),
        'community_info': result.get('community_info', []),
        'baike_lemma_id': result.get('baike_lemma_id'),
        'baike_lemma_list': result.get('baike_lemma_list', []),
        'llm_timings': result.get('llm_timings', []),
    }

    if status['status'] == 'error':
        response['error'] = status.get('error', '')

    return jsonify(response)


@app.route('/select_top_houses', methods=['POST'])
def select_top_houses():
    """
    手动触发 TOP5 精选房源（三种版本）

    请求体 JSON:
        { 
            "task_id": "uuid-string", 
            "version": "balance|price_sensitive|quality" 
        }
    
    version 参数说明:
        - balance: 综合兼顾性价比版本
        - price_sensitive: 价格敏感优先版本（侧重低价、划算、省钱）
        - quality: 品质居住版本（侧重居住、户型、楼层、装修）

    返回:
        { "success": true, "top_houses": [...], "selection_version": "...", "llm_timings": [...] }
    """
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        version = data.get('version', 'balance')

        if not task_id:
            return jsonify({'error': '缺少 task_id'}), 400

        # 获取之前的搜索结果
        status = get_task_status(task_id)
        if not status or status['status'] != 'done':
            return jsonify({'error': '任务不存在或未完成'}), 400

        result = get_task_result(task_id)
        
        # 调试信息
        print(f"📋 TOP5精选 - task_id: {task_id}")
        print(f"📋 TOP5精选 - result keys: {list(result.keys()) if result else 'None'}")
        print(f"📋 TOP5精选 - house_comments count: {len(result.get('house_comments', [])) if result else 0}")
        print(f"📋 TOP5精选 - leyoujia_results count: {len(result.get('leyoujia_results', [])) if result else 0}")
        print(f"📋 TOP5精选 - community_name: {result.get('community_name', 'None')}")
        
        # 创建状态对象用于 TOP5 精选
        from leyoujia_ana import HouseFinderState
        state = HouseFinderState(
            community_name=result.get('community_name', ''),
            house_comments=result.get('house_comments', []),
            leyoujia_results=result.get('leyoujia_results', []),
            community_info=result.get('community_info', []),
            task_id=task_id
        )

        # 根据版本选择对应的函数
        if version == 'price_sensitive':
            top_result = agent.select_top_houses_price_sensitive(state)
        elif version == 'quality':
            top_result = agent.select_top_houses_quality(state)
        else:
            top_result = agent.select_top_houses_balance(state)

        return jsonify({
            'success': True,
            'top_houses': top_result.get('top_houses', []),
            'selection_version': top_result.get('selection_version', ''),
            'llm_timings': top_result.get('llm_timings', []),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
