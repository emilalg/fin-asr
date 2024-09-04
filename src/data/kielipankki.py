from torch.utils.data import Dataset
import glob
import librosa
import logging

# https://clarino.uib.no/comedi/editor/lb-2022052002
# https://www.kielipankki.fi/download/fi-parliament-asr/fi-parliament-asr-v2/
class KielipankkiDataset(Dataset):

    def __init__(self, data_dir='data/kielipankki', mode='train'):
        
        self.data = []

        match mode:
            case 'train' : 
                # iterate over audio files
                for f in glob.glob(f'{data_dir}/**/*.wav', recursive=True):
            
                    # dont include dev test in train set
                    if 'dev-test' not in f:

                        transcript_file = f.replace('.wav', '.trn')

                        self.data.append({
                            'audio' : f,
                            'transcript' : transcript_file
                        })
            

            case 'validation':
                pass
            

            case 'test':
                pass


            case _:
                raise Exception('Dataset mode not properly set')
        
        logging.info(f'Kielipankki dataset length : {len(self.data)}')
    



    def __len__(self):
        return len(self.data)
                



    def __getitem__(self, idx):
        item = self.data[idx]

        audio, sr = librosa.load(item['audio'])

        transcript = ''
        with open(item['transcript']) as f:
            transcript = f.read()
            f.close()

        # same formatting as comon_voice_17
        return {
            'audio' : {
                'array' : audio,
                'sampling_rate' : sr
            },
            'sampling_rate' : sr,
            'sentence' : transcript
        }
