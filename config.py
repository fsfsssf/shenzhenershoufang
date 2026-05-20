#!/usr/bin/env python3
"""
配置文件 — 本项目所有 API 密钥和配置的集中存储

本模块通过 Config 类统一管理以下服务的 API 密钥：
  - 百度百科 API (百度千帆 AppBuilder)
  - 大语言模型 API (DeepSeek，兼容 OpenAI SDK)
  - 乐有家二手房平台 API

架构设计：
  所有模块通过 get_xxx() 函数获取密钥，不直接访问 Config 类，
  这样未来若需改为从环境变量或加密存储读取，只需修改本文件。

⚠️ 安全提示：
  API 密钥通过环境变量读取，请勿将真实密钥直接写入代码。
  复制 .env.example 为 .env 并填入真实密钥。
"""

import os


class Config:
    """
    配置类 — 存储所有 API 密钥

    字段说明：
      BAIDU_API_KEY   : 百度千帆 API Key，用于调用百科词条 search/content 接口
      LLM_API_KEY     : DeepSeek API Key，用于调用大语言模型
      LLM_API_BASE_URL: DeepSeek API 基础 URL，兼容 OpenAI SDK 格式
      LEYOUJIA_API_KEY: 乐有家开放平台 API Key，用于查询房源和小区信息
    """

    # 百度百科 API — 从环境变量读取
    BAIDU_API_KEY = os.environ.get("BAIDU_API_KEY", "")

    # 大语言模型 API — 支持 OpenAI SDK 兼容的任何服务商
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "deepseek-chat")

    # 乐有家二手房平台 API — 从环境变量读取
    LEYOUJIA_API_KEY = os.environ.get("LEYOUJIA_API_KEY", "")


# 全局配置实例 — 模块级单例
config = Config()


def get_baike_api_key() -> str:
    """获取百度百科 API 密钥"""
    return config.BAIDU_API_KEY


def get_llm_api_key() -> str:
    """获取 LLM API 密钥"""
    return config.LLM_API_KEY


def get_llm_api_base_url() -> str:
    """获取 LLM API 基础 URL"""
    return config.LLM_API_BASE_URL


def get_llm_model_name() -> str:
    """获取 LLM 模型名称"""
    return config.LLM_MODEL_NAME


def get_leyoujia_api_key() -> str:
    """获取乐有家 API 密钥"""
    return config.LEYOUJIA_API_KEY
