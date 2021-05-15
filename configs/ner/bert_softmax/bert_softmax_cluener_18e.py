_base_ = [
    '../../_base_/schedules/schedule_adadelta_18e.py',
    '../../_base_/default_runtime.py'
]

categories = [
    'address', 'book', 'company', 'game', 'government', 'movie', 'name',
    'organization', 'position', 'scene'
]

test_ann_file = 'data/cluener_public/dev.json'
train_ann_file = 'data/cluener_public/train.json'
vocab_file = 'data/cluener_public/vocab.txt'
pretrained = 'checkpoints/bert_pretrain.pth'
max_len = 128
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(type='LineJsonParser', keys=['text', 'label']))

ner_convertor = dict(
    type='NerConvertor',
    annotation_type='bio',
    vocab_file=vocab_file,
    categories=categories,
    max_len=max_len)

test_pipeline = [
    dict(type='NerTransform', label_convertor=ner_convertor, max_len=max_len),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids', 'attention_mask',
            'token_type_ids'
        ]),
    dict(type='ToTensorNER')
]

train_pipeline = [
    dict(type='NerTransform', label_convertor=ner_convertor, max_len=max_len),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'texts', 'img', 'labels', 'input_ids', 'attention_mask',
            'token_type_ids'
        ]),
    dict(type='ToTensorNER')
]
dataset_type = 'NerDataset'
img_prefix = ''

train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_ann_file,
    loader=loader,
    pipeline=train_pipeline,
    test_mode=False)

test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_ann_file,
    loader=loader,
    pipeline=test_pipeline,
    test_mode=True)
data = dict(
    samples_per_gpu=24, workers_per_gpu=2, train=train, val=test, test=test)

evaluation = dict(interval=1, metric='acc')

model = dict(
    type='NerClassifier',
    pretrained=pretrained,
    encoder=dict(type='BertEncoder', max_position_embeddings=512),
    decoder=dict(type='FCDecoder'),
    loss=dict(type='MaskedCrossEntropyLoss'),
    label_convertor=ner_convertor)

test_cfg = None
