import datasets


class HuggingFaceDataset():

    def __init__(self, data_dir='data/hf') -> None:
        # setup hf token through the huggingface cli

        self.train_dataset = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",  
            "fi",
            split='train',
            cache_dir=data_dir,
            token=True,
            trust_remote_code=True
        )

        self.val_dataset = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",  
            "fi",
            split='validation',
            cache_dir=data_dir,
            token=True,
            trust_remote_code=True
        )

        self.test_dataset = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",  
            "fi",
            split='test',
            cache_dir=data_dir,
            token=True,
            trust_remote_code=True
        )





    