from dataclasses import dataclass, field

@dataclass
class Config:
    train_path: str = field(
        default = "../data/train_7000_auto.json",
        metadata = { "help": "训练数据位置" }
    )
    dev_path: str = field(
        default = "../data/valid_1500_auto.json",
        metadata = { "help": "验证数据位置" }
    )
    id2label_path: str = field(
        default = "../data/id2label.txt",
        metadata = { "help": "标签集合的列表存放位置" }
    )
    pretrained_model_path: str = field(
        default = "/tf/FangGexiang/2.SememeV2/pretrained_model/chinese-bert-wwm-ext/",
        metadata = { "help": "预训练模型的存放位置" }
    )
    batch_size: int = field(
        default = 10,
        metadata = { "help": "批处理大小" }
    )
    max_len: int = field(
        default = 512,
        metadata = { "help": "每句最长大小" }
    )
    device: str = field(
        default = "cuda:0",
        metadata = { "help": "GPU" }
    )
    lr: float = field(
        default = 5e-5,
        metadata = { "help": "BERT的学习率" }
    )
    epoches: int = field(
        default = 10,
        metadata = { "help": "迭代次数" }
    )
    save_path: str = field(
        default = "../../model_saved/",
        metadata = { "help": "模型保存位置" }
    )
    vocab_size: int = field(
        default = 21128,
        metadata = { "help": "词表大小" }
    )
    logger_path: str = field(
        default = "./logger/training_log.txt",
        metadata = { "help": "模型保存位置" }
    )
    weight_decay: float = field(
        default = 1e-4,
        metadata = { "help": "权重衰减项，防止过拟合的一个参数" }
    )
    bert_lr: float = field(
        default = 5e-5,
        metadata = { "help": "BERT的基础学习率" }
    )
    learning_rate: float = field(
        default = 1e-4,
        metadata = { "help": "其他的学习率" }
    )