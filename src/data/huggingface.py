import logging
import datasets


class HuggingFaceDataset():

    def __init__(self, token, predl, data_dir='data/hf') -> None:
        # setup hf token through the huggingface cli

        if not predl:

            if token == '':
                raise Exception('Hf token not provided')

            self.train_dataset = datasets.load_dataset(
                "mozilla-foundation/common_voice_17_0",  
                "fi",
                split='train',
                cache_dir=data_dir,
                token=token,
                trust_remote_code=True
            )

            self.val_dataset = datasets.load_dataset(
                "mozilla-foundation/common_voice_17_0",  
                "fi",
                split='validation',
                cache_dir=data_dir,
                token=token,
                trust_remote_code=True
            )

            self.test_dataset = datasets.load_dataset(
                "mozilla-foundation/common_voice_17_0",  
                "fi",
                split='test',
                cache_dir=data_dir,
                token=token,
                trust_remote_code=True
            )
        
        else:
            
            try:
                dataset = datasets.load_from_disk(dataset_path=data_dir)
                self.train_dataset = dataset['train']
                self.val_dataset = dataset['validation']
                self.test_dataset = dataset['test']
            except:
                dataset = datasets.load_from_disk(dataset_path=data_dir)
                logging.error(f'Error in loading from disk {dataset.keys()}')
            




    