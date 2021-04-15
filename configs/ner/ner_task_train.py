_base_ = [
    '../../../configs/_base_/schedules/schedule_adadelta_8e.py',
    '../../../configs/_base_/default_runtime.py',
    'ner_dataset_train.py',
    'ner_model_train.py'
]