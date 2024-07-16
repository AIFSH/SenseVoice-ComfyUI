import os
now_dir = os.path.dirname(os.path.abspath(__file__))

import torchaudio
from funasr import AutoModel
# from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope import snapshot_download

pre_model_dir = os.path.join(now_dir,"pretrianed_models","SenseVoiceSmall")
snapshot_download(model_id="iic/SenseVoiceSmall",local_dir=pre_model_dir)

class SenseVoiceNode:

    def __init__(self) -> None:
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "batch_size_s":("INT",{
                    "default": 0
                })
            }
        }
    
    RETURN_TYPES = ("TEXT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    CATEGORY = "AIFSH_SenseVoice"

    def generate(self,audio,batch_size_s):
        if self.model is None:
            self.model = AutoModel(model=pre_model_dir,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    trust_remote_code=True, device="cuda:0")
        audio_data = audio["waveform"].squeeze(0)
        audio_rate = audio['sample_rate']
        if audio_rate != 16000:
            audio_data = torchaudio.transforms.Resample(audio_rate,16000)(audio_data)
        
        # audio_data = torchaudio.compliance.kaldi.fbank(audio_data)
        audio_data = audio_data.numpy()
        # print(audio_data.shape)
        res = self.model.generate(
            input=[audio_data],
            cache={},
            language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            batch_size_s=batch_size_s, 
        )
        print(res[0]["text"])
        return (res[0]["text"],)

class ShowTextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sense_voice_output":("TEXT",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "encode"
    OUTPUT_NODE = True
    CATEGORY = "AIFSH_SenseVoice"

    def encode(self,sense_voice_output,text):
        return {"ui":{"text":[sense_voice_output]}}


WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "ShowTextNode":ShowTextNode,
    "SenseVoiceNode": SenseVoiceNode
}
