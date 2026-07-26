transformer/
├── __init__.py        # 包入口（必须）
├── norm.py            # BatchNorm/LayerNorm
├── linear.py          # 全连接层（Linear）
├── embedding.py       # 词嵌入 + 位置编码
├── attention.py       # 自注意力、掩码注意力、多头注意力
├── ffn.py             # 前馈网络  Linear + 激活
├── encoder.py         # 编码器层  注意力 + FFN + 残差 + Norm
├── decoder.py         # 解码器层  掩码自注意力 + 交叉注意力 + FFN
├── transformer.py     # 总模型（Encoder+Decoder）
└── utils.py           # 掩码、工具函数