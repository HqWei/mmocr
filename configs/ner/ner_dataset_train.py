img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
ann_file="/data/project/MMOCR/new/mmocr/demo/dev.json"
loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['text', 'label']))

test_pipeline = [
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids','attention_mask','token_type_ids'
        ])
]

train_pipeline = [
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids', 'attention_mask', 'token_type_ids'
        ])
]

pipeline=test_pipeline


dataset_type = 'NerDataset'
img_prefix = ''
train_ann_file="/data/project/MMOCR/new/mmocr/demo/train.json"

train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_ann_file,
    loader=loader,
    pipeline=train_pipeline,
    test_mode=False,
    vocab_file='/data/project/NER/BERT-NER-Pytorch/prev_trained_model/bert-base/vocab.txt'
)

test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=ann_file,
    loader=loader,
    pipeline=test_pipeline,
    test_mode=True,
    vocab_file='/data/project/NER/BERT-NER-Pytorch/prev_trained_model/bert-base/vocab.txt'
)
#samples_per_gpu=32,
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train]),
    val=dict(type='ConcatDataset', datasets=[test]),
    test=dict(type='ConcatDataset', datasets=[test]))

evaluation = dict(interval=1, metric='acc')
