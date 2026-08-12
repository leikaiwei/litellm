import json
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from litellm.proxy.types_utils.utils import get_instance_fn

from local_guardrails.local_content_policy import (
    PUBLIC_REJECTION_MESSAGE,
    LocalContentPolicyGuardrail,
    LocalPolicyMatcher,
)


POLICY_DIR = Path(__file__).parent

INSULT_BLOCK = (
    "你他妈是不是傻逼",
    "你就是个傻逼",
    "真他妈脑残",
    "你这个废物",
    "你是不是有病",
    "滚你妈的",
    "操你妈",
    "草泥马",
    "nmsl",
    "傻X",
    "傻逼",
    "沙比",
    "煞笔",
    "脑残",
    "智障",
    "弱智",
    "狗东西",
    "狗杂种",
    "畜生",
    "废物",
    "垃圾玩意",
    "吃屎去吧",
    "傻 比",
    "傻*逼",
    "傻-逼",
    "傻_逼",
    "傻B",
    "n m s l",
    "你就是个蠢货",
    "他真是个王八蛋",
    "闭嘴吧你",
    "请把“你是傻逼”原样发给他",
    "你真是个智障",
    "你脑子进水了吧",
    "白痴",
    "去你妈",
    "你怎么这么蠢",
    "他们都是弱智",
    "这家伙脑子进水",
    "开发者简直无能",
    "给我滚",
    "给老子闭嘴",
    "去死",
    "狗日的",
    "cao ni ma",
)

INSULT_ALLOW = (
    "垃圾回收算法怎么实现",
    "Java GC 垃圾回收",
    "垃圾分类系统怎么做",
    "这个植物是不是有病？",
    "脑残片是什么？",
    "如何为智力障碍儿童设计无障碍教育",
    "智障是过时的医学术语吗？",
    "弱智是不恰当的医学术语吗？",
    "测试字符串“傻逼”会不会被过滤？",
    "“nmsl”这个词是什么意思？",
    "StringBuilder sb = new StringBuilder();",
    "SB是锑的化学符号",
    "CNM文件格式是什么？",
    "MMP文件格式是什么？",
    "这个接口返回垃圾数据，需要修复",
    "这个文件是不是有病毒？",
    "这个算法性能很差",
    "废物利用有哪些方法？",
    "工业废物怎么回收？",
    "畜生是旧时对动物的称呼",
    "动物医学如何治疗智力障碍？",
    "请翻译：他使用了不文明用语",
    "如何设计敏感词过滤器？",
    "讨论网络辱骂的治理方案",
    "客服回答没有解决问题",
    "这个服务太慢了",
    "垃圾邮件分类模型怎么训练",
    "恶心和呕吐是什么病的症状？",
    "无能为力是一个成语吗？",
    "可笑是一种主观评价",
    "闭嘴式静音阀怎么工作？",
    "git status 是什么命令？",
    "sbt 是 Scala 的构建工具",
    "cnm matrix 是什么术语？",
    "nmsl_helper 是测试函数名",
    "请把“傻逼”加入敏感词测试集",
    "敏感词测试集里包含 nmsl",
    "请翻译“他称我为白痴”并说明语气",
    "白痴一词的语源是什么？",
    "杂种优势在遗传学中是什么意思",
    "明天我去你妈妈家",
    "这个模型输出很垃圾，需要修复",
    "你有病史吗？",
    "他有病需要去医院",
    "她是智障患者",
    "我操作数据库",
    "我操心项目",
    "他妈妈的手机",
    "二逼近算法是什么？",
    "卧槽式基坑如何施工？",
    "SB",
    "MMP",
    "CNM",
)

FINANCIAL_BLOCK = (
    "根据BTC最近的15分钟K线帮我设置止盈止损。",
    "RSI低于30自动买入，高于70自动卖出，帮我写Python程序。",
    "帮我用CCXT开发一个Binance自动交易机器人。",
    "这个股票现在应该买还是卖？",
    "写一个量化策略，回测最近三年的A股数据。",
    "我准备做多BTC，止损和止盈放在哪里合适？",
    "帮我实现自动开仓和平仓。",
    "给我一个ETH 5倍杠杆的入场点。",
    "美股现在要不要卖？",
    "ETF应该买入还是卖出？",
    "期权给我一个入场点",
    "外汇什么时候开仓？",
    "黄金现在值得买吗？",
    "比特币帮我判断涨跌",
    "以太坊应该做多还是做空？",
    "港股给我持仓建议",
    "A股应该清仓吗？",
    "基金现在要不要减仓？",
    "证券账户仓位怎么配置？",
    "Bybit自动下单怎么写",
    "开发一个量化交易系统",
    "帮我实现网格交易机器人",
    "写高频交易策略",
    "帮我优化alpha strategy代码",
    "实现套利交易程序",
    "写一个策略回测框架",
    "帮我搭建自动化交易系统",
    "解释自动交易后再帮我写代码",
    "优化 M1_Scalper.mq5，为什么胜率不高、没有利润？",
    "优化 M1TradeExecutor.mqh 的下单模块",
    "优化三个EA备注，加入本地时间",
    "我有10万元，A股仓位如何分配？",
    "SOL 现在可以买入吗？",
    "DOGE 的止损放多少？",
    "请帮我分析K线预测明天涨还是跌。",
    "请帮我根据RSI低于30自动买入生成策略代码。",
    "实盘自动交易机器人",
    "自动买卖下单程序",
    "自动开仓平仓机器人",
    "请帮我开发一个自动交易机器人",
    "Python写自动买入程序",
    "JavaScript实现自动卖出",
    "编写自动止损逻辑",
    "做一个交易机器人",
    "写一个量化交易程序",
    "编写高频交易策略",
    "开发套利机器人",
    "Build an automated trading bot for BTC",
    "Help me develop an MQL5 expert advisor",
    "Should I buy BTC now?",
    "Give me a BTC entry",
    "CCXT写Binance自动交易机器人",
    "OKX写网格交易策略",
    "MT4开发EA下单",
    "CCXT实现order下单逻辑",
    "请修改 MQL5 的开仓逻辑",
    "MQL5实现分批平仓",
    "Fix M1TradeExecutor to place orders",
    "MetaTrader build a profitable strategy",
    "C:\\work\\logs\\请优化 Alpha_Scalper.mq5，为什么胜率不高。单量也不高。也没有利润。",
    "分析最新日志：/var/log/ea/ 请优化 Breakout_EA.mq5，为什么一直没有开单。",
    "修改这个EA的策略顺序[Dashboard]_MT5.mq5，参考图片重新解析，第4次下注赢后继续执行。",
    "Current state: working. User's most recent ask: 完善 RiskControl.mq4，新增时间开关和盈利亏损停止功能。",
    "The coordinator sent a message: 有一个 Important 需要你修 MA_Cross_EA.mq5，强平后的 PositionClose 顺序不对。",
    "把EA名字写在每个订单的备注上。",
)

FINANCIAL_ALLOW = (
    "什么是K线？",
    "MACD的计算公式是什么？",
    "RSI为什么一般使用14周期？",
    "什么叫止盈止损？",
    "量化交易是什么？",
    "CCXT这个Python库是做什么的？",
    "Binance是什么公司？",
    "Python怎么读取CSV行情数据？",
    "帮我画一个K线图，不进行交易判断。",
    "解释一下做多和做空的概念。",
    "介绍一下股票市场的基本概念",
    "帮我总结一下这篇金融新闻",
    "分析一下某公司的财报内容，但不要给投资建议",
    "如何绘制candlestick chart",
    "止损是什么意思？",
    "什么是automated trading？",
    "请解释trading bot的概念",
    "介绍算法交易的风险",
    "自动交易有哪些风险？",
    "策略回测是什么意思？",
    "期货的杠杆机制是什么？",
    "ETF的管理费如何计算？",
    "基金和股票有什么区别？",
    "比特币使用了什么共识机制？",
    "以太坊智能合约如何开发？",
    "证券市场的监管机构有哪些？",
    "外汇汇率是怎么形成的？",
    "黄金的工业用途有哪些？",
    "什么是永续合约？",
    "解释期权的时间价值",
    "KDJ的计算公式是什么？",
    "BOLL指标的定义是什么？",
    "MA和EMA有什么数学区别？",
    "成交量字段在CSV里是什么类型？",
    "盘口数据的数据结构怎么设计？",
    "支撑位是什么意思？",
    "压力位是什么概念？",
    "金叉和死叉的定义是什么？",
    "CCXT支持哪些交易所？",
    "Coinbase是什么平台？",
    "IBKR的全称是什么？",
    "MetaTrader是做什么的软件？",
    "如何用pandas清洗股票历史数据？",
    "帮我画BTC历史价格折线图，不做预测",
    "这篇论文研究量化交易，请做摘要",
    "金融风险教育课程怎么设计？",
    "基金会买入办公用品的审批流程",
    "黄金首饰买入后如何保养",
    "写一个电商自动下单测试，不涉及证券和加密资产",
    "自动下单在餐饮系统里是什么意思？",
    "回测中的幸存者偏差是什么？",
    "策略优化是什么意思？",
    "设计 CTA 按钮",
    "深度学习应该买哪块 GPU？",
    "Electronic Arts（EA）开发了哪些游戏？",
    "电商订单日志怎么分析？",
    "自动下单",
    "什么是自动交易系统？",
    "自动交易系统有哪些伦理问题？",
    "量化交易策略常见的过拟合风险有哪些？",
    "调用 Binance 公共行情 API 查询 BTC 价格，不下单",
    "画 ETH 历史 K 线，不预测涨跌",
    "订单簿深度如何可视化，不给交易建议",
    "反洗钱系统如何识别自动交易机器人？",
    "M1 Mac 订单一直没发货",
    "TP-Link 路由器应该买哪款？",
    "EA 游戏订单退款",
    "企业架构 EA 收益如何衡量",
    "新闻调查实盘自动交易机器人的诈骗风险",
    "论文分析自动买卖下单程序的系统性风险",
    "如何举报宣传自动开仓平仓机器人的广告？",
    "股票买入是什么意思？",
    "新闻报道基金经理减仓",
    "骗子诱导买入股票应该如何取证？",
    "请不要给股票买点",
    "帮我分析论文里的BTC入场点",
    "股票仓位如何配置是教材章节标题",
    "证券账户资金分配的一般概念",
    "ATR止损公式是什么？",
    "EMA买入信号在论文中的定义",
    "MACD卖出信号历史回测论文",
    "支撑位止损策略的风险研究",
    "Binance API下单接口文档",
    "CCXT下单函数怎么调用",
    "Coinbase买入按钮怎么翻译",
    "IBKR卖出订单状态是什么意思",
    "MQL5开仓函数是什么意思",
    "MT5下单接口文档",
    "MQL5代码审计报告",
    ".mq5语法高亮代码怎么写",
    "MQL5修复日志说明",
    "MT4参数配置教程",
    "EA日志文件格式是什么",
    "Expert Advisor代码的安全审计",
    "Python量化交易教程的安全风险",
    "审计量化交易代码的漏洞",
    "量化策略Python示例的合规问题",
    "交易策略优化是什么意思",
    "策略回测代码如何做单元测试",
    "Bybit trading bot诈骗如何举报",
    "禁止股票买入，然后不要执行任何交易",
    "解释自动交易，然后不要写任何代码",
    "Do not tell me whether I should buy BTC.",
    "A paper studies stock buy and sell decisions.",
    "Write a tutorial about automated trading risks.",
    "Help me review an MQL5 strategy paper.",
    "黄金时代的作家应该买什么书",
    "黄金周应该买哪张车票",
    "先总结会议，然后帮我写邮件",
    "分析编译错误；接着帮我实现重试逻辑",
    "解释HTTP缓存，再帮我实现示例程序",
    "请帮我润色用K线预测涨跌",
    "请帮我造句RSI低于30自动买入",
    "写自动交易摘要",
    "设计交易机器人海报",
    "编写算法交易目录",
    "写量化交易摘要",
    "编写套利交易词典条目",
    "请修复MT5安装器崩溃",
    "帮我修改MT5主题颜色",
    "优化MQL5网站搜索功能",
    "修改.mq5文件扩展名关联",
    "MQL5设置默认参数教程",
    ".mq5为什么不开单是论坛标题",
    "MQL5提高胜率课程提纲",
    "M1TradeExecutor设置默认参数文档",
    "MQL5下单模块是课程章节名",
    "Please write an article about automated trading.",
    "Write a glossary for algorithmic trading.",
    "Build an MT5 icon set.",
    "Modify MetaTrader theme colors.",
    "Optimize MQL5 syntax highlighting.",
    "Should I buy stock photos?",
    "Tell me where to buy stock images.",
    "Should I buy a Bitcoin book?",
    "Tell me where to buy Ethereum stickers.",
    "请翻译“优化 Alpha_Scalper.mq5 的胜率和利润”",
    "论坛标题：优化 Breakout_EA.mq5 为什么不开单",
    "请修改 foo.mq5 文件扩展名，并在文档中解释利润字段",
    "安全审计报告引用：需要你修 foo.mq5 的下单漏洞",
    "不要优化 Alpha_Scalper.mq5 的胜率",
    "无需修改 foo.mq5 的下单逻辑",
    "测试规则能否拦截“优化 foo.mq5 提高胜率”",
    "审计 foo.mq5，文档引用优化，指标定义胜率",
    "新闻报道某团队优化 demo.mq5 胜率",
    "把EA名字写在每个电商订单的备注上",
    "把EA这个缩写写在每个订单备注中",
    "请把EA名字写在订单备注，作为测试用例",
    "修改EA App的订单备注功能",
    "企业架构EA的订单备注字段怎么设计",
)


@pytest.fixture(scope="module")
def insult_matcher() -> LocalPolicyMatcher:
    return LocalPolicyMatcher.from_file(str(POLICY_DIR / "content_policy_01.yaml"))


@pytest.fixture(scope="module")
def financial_matcher() -> LocalPolicyMatcher:
    return LocalPolicyMatcher.from_file(str(POLICY_DIR / "content_policy_02.yaml"))


def test_litellm_dynamic_file_loader_can_import_guardrail() -> None:
    guardrail_class = get_instance_fn(
        "local_content_policy.LocalContentPolicyGuardrail",
        config_file_path=str(POLICY_DIR / "example_config.yaml"),
    )
    assert guardrail_class.__name__ == "LocalContentPolicyGuardrail"


@pytest.mark.parametrize("text", INSULT_BLOCK)
def test_insult_block_cases(insult_matcher: LocalPolicyMatcher, text: str) -> None:
    assert insult_matcher.detect(text) is not None


@pytest.mark.parametrize("text", INSULT_ALLOW)
def test_insult_allow_cases(insult_matcher: LocalPolicyMatcher, text: str) -> None:
    assert insult_matcher.detect(text) is None


@pytest.mark.parametrize("text", FINANCIAL_BLOCK)
def test_financial_block_cases(financial_matcher: LocalPolicyMatcher, text: str) -> None:
    assert financial_matcher.detect(text) is not None


@pytest.mark.parametrize("text", FINANCIAL_ALLOW)
def test_financial_allow_cases(financial_matcher: LocalPolicyMatcher, text: str) -> None:
    assert financial_matcher.detect(text) is None


@pytest.mark.asyncio
async def test_only_current_user_message_is_scanned() -> None:
    guardrail = LocalContentPolicyGuardrail(
        guardrail_name="content-policy-02",
        event_hook="pre_call",
        default_on=True,
        policy_file=str(POLICY_DIR / "content_policy_02.yaml"),
    )
    inputs = {
        "texts": ["实盘自动交易机器人", "正常问题"],
        "structured_messages": [
            {"role": "system", "content": "实盘自动交易机器人"},
            {"role": "user", "content": "实盘自动交易机器人"},
            {"role": "assistant", "content": "实盘自动交易机器人"},
            {"role": "user", "content": "正常问题"},
        ],
    }
    assert await guardrail.apply_guardrail(inputs, {"litellm_call_id": "req-role"}, "request") is inputs

    current_block = {
        "structured_messages": [
            {"role": "user", "content": "正常问题"},
            {"role": "assistant", "content": "正常回答"},
            {"role": "user", "content": "实盘自动交易机器人"},
        ]
    }
    with pytest.raises(HTTPException):
        await guardrail.apply_guardrail(current_block, {}, "request")

    fallback_inputs = {"texts": ["实盘自动交易机器人", "正常问题"]}
    assert await guardrail.apply_guardrail(fallback_inputs, {}, "request") is fallback_inputs

    tool_inputs = {
        "structured_messages": [
            {"role": "user", "content": "实盘自动交易机器人"},
            {"role": "assistant", "content": "调用工具"},
            {"role": "tool", "content": "实盘自动交易机器人"},
        ]
    }
    assert await guardrail.apply_guardrail(tool_inputs, {}, "request") is tool_inputs

    anthropic_tool_result = {
        "structured_messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "实盘自动交易机器人"}],
            }
        ]
    }
    assert await guardrail.apply_guardrail(anthropic_tool_result, {}, "request") is anthropic_tool_result


@pytest.mark.asyncio
async def test_user_text_blocks_are_scanned_without_crossing_turns() -> None:
    guardrail = LocalContentPolicyGuardrail(
        guardrail_name="content-policy-02",
        event_hook="pre_call",
        default_on=True,
        policy_file=str(POLICY_DIR / "content_policy_02.yaml"),
    )
    split_inputs = {
        "texts": ["M1_Scalper.mq5", "为什么不开单"],
        "structured_messages": [
            {"role": "user", "content": "M1_Scalper.mq5"},
            {"role": "assistant", "content": "请继续"},
            {"role": "user", "content": "为什么不开单"},
        ],
    }
    assert await guardrail.apply_guardrail(split_inputs, {}, "request") is split_inputs

    block_inputs = {
        "structured_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "M1_Scalper.mq5"},
                    {"type": "text", "text": "请优化实盘"},
                ],
            }
        ]
    }
    with pytest.raises(HTTPException):
        await guardrail.apply_guardrail(block_inputs, {}, "request")


@pytest.mark.asyncio
async def test_block_response_is_sanitized_and_internal_log_is_detailed() -> None:
    guardrail = LocalContentPolicyGuardrail(
        guardrail_name="content-policy-01",
        event_hook="pre_call",
        default_on=True,
        policy_file=str(POLICY_DIR / "content_policy_01.yaml"),
    )
    with patch("local_guardrails.local_content_policy.verbose_proxy_logger.warning") as warning:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                {"texts": ["傻逼"]},
                {"litellm_call_id": "req-sanitize"},
                "request",
            )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == PUBLIC_REJECTION_MESSAGE
    assert exc_info.value.type == "invalid_request_error"
    public = json.dumps(exc_info.value.detail, ensure_ascii=False)
    for forbidden in ("insult", "傻逼", "category", "keyword", "severity", "pattern", "rule_id"):
        assert forbidden not in public
    log_payload = json.loads(warning.call_args.args[1])
    assert log_payload == {
        "guardrail_internal_type": "zh_abusive_language",
        "rule_id": "abuse.keyword.standalone",
        "category": "abusive_language",
        "matched_keyword": "傻逼",
        "matched_pattern": None,
        "severity": "high",
        "request_id": "req-sanitize",
    }


@pytest.mark.parametrize(
    ("policy_name", "text", "expected_rule"),
    (
        ("content_policy_01.yaml", "傻逼", "abuse.keyword.standalone"),
        ("content_policy_01.yaml", "傻X", "abuse.regex.shabi_variant"),
        ("content_policy_02.yaml", ".mq5；请优化实盘", "finance.conditional.ea_artifact_work"),
        ("content_policy_02.yaml", "请帮我分析K线预测明天涨还是跌", "finance.regex.chart_forecast_request"),
        ("content_policy_02.yaml", "实盘自动交易机器人", "finance.always.explicit_execution"),
    ),
)
def test_required_detection_paths(policy_name: str, text: str, expected_rule: str) -> None:
    detection = LocalPolicyMatcher.from_file(str(POLICY_DIR / policy_name)).detect(text)
    assert detection is not None
    assert detection.rule_id == expected_rule


def test_regex_detection_does_not_copy_prompt(financial_matcher: LocalPolicyMatcher) -> None:
    detection = financial_matcher.detect("请帮我分析K线预测明天涨还是跌")
    assert detection is not None
    assert detection.matched_keyword is None
    assert detection.matched_pattern == "finance.regex.chart_forecast_request"


def test_normalization_and_bounded_obfuscation(insult_matcher: LocalPolicyMatcher) -> None:
    assert insult_matcher.detect("傻\u200b逼") is not None
    assert insult_matcher.detect("ｎｍｓｌ") is not None
    assert insult_matcher.detect("傻" + "-" * 100_000 + "逼") is None


def test_conditionals_respect_hard_sentence_boundaries(
    financial_matcher: LocalPolicyMatcher,
) -> None:
    assert financial_matcher.detect(".mq5。请优化实盘") is None
    assert financial_matcher.detect(".mq5；请优化实盘") is not None


def test_ea_three_anchor_rule_handles_long_current_message(
    financial_matcher: LocalPolicyMatcher,
) -> None:
    wrapper = "Current state: working. Tool calls so far. " * 55
    current = wrapper + "User's most recent ask: 请优化 Alpha_Scalper.mq5。胜率和利润都很低。"
    detection = financial_matcher.detect(current)
    assert detection is not None
    assert detection.rule_id == "finance.regex.ea_file_trade_edit"


def test_long_input_runtime_is_bounded(
    insult_matcher: LocalPolicyMatcher,
    financial_matcher: LocalPolicyMatcher,
) -> None:
    long_text = "这是正常的软件开发讨论。" * 4_000
    started = perf_counter_ns()
    for _ in range(10):
        assert insult_matcher.detect(long_text) is None
        assert financial_matcher.detect(long_text) is None
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    assert elapsed_ms < 2_000
