"""ViT-L/14 + LinCIR projection from the authors' checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPImageProcessor
from huggingface_hub import hf_hub_download

MODEL='openai/clip-vit-large-patch14'
REVISION='32bd64288804d66eefd0ccbe215aa642df71cc41'
PHI_REVISION='5118df4683de6efa09f8e5336d86ca626ed21c44'

class Phi(nn.Module):
    def __init__(self,dim,hidden,out):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(0),nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(0),nn.Linear(hidden,out))
    def forward(self,x): return self.layers(x)

class LinCIR:
    def __init__(self, device='auto'):
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA không khả dụng')
        if device=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS không khả dụng')
        self.device=torch.device(device); self.dtype=torch.float16 if device=='cuda' else torch.float32
        self.processor=CLIPImageProcessor.from_pretrained(MODEL,revision=REVISION)
        self.tokenizer=CLIPTokenizer.from_pretrained(MODEL,revision=REVISION)
        if self.tokenizer.encode('$',add_special_tokens=False)!=[259]: raise ValueError('Token $ không tương thích')
        kwargs=dict(revision=REVISION,torch_dtype=self.dtype,attn_implementation='eager')
        self.vision=CLIPVisionModelWithProjection.from_pretrained(MODEL,**kwargs)
        self.text=CLIPTextModelWithProjection.from_pretrained(MODEL,**kwargs)
        self.phi=Phi(self.text.config.projection_dim,4*self.text.config.projection_dim,self.text.config.hidden_size)
        p=hf_hub_download('navervision/zeroshot-cir-models','lincir_large.pt',revision=PHI_REVISION)
        self.phi.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['Phi'])
        for m in (self.vision,self.text,self.phi): m.to(device=self.device,dtype=self.dtype).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_images(self, images):
        pixels=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device,self.dtype)
        # Return raw embedding: projection Phi requires unnormalised image features.
        return self.vision(pixel_values=pixels).image_embeds.float().cpu()

    @torch.inference_mode()
    def rank(self,reference,condition,features):
        token=self.phi(reference.unsqueeze(0).to(self.device,self.dtype))
        ids=self.tokenizer(f'a photo of $ that {condition}',padding='max_length',max_length=77,truncation=True,return_tensors='pt')['input_ids'].to(self.device)
        mask=ids.eq(259)
        if mask.sum().item()!=1: raise ValueError('Pseudo-token không hợp lệ')
        def insert(module,inputs,output):
            return torch.where(mask.unsqueeze(-1),token.to(output.dtype).unsqueeze(1),output)
        handle=self.text.text_model.embeddings.token_embedding.register_forward_hook(insert)
        try: query=self.text(input_ids=ids).text_embeds.float().cpu()
        finally: handle.remove()
        scores=F.normalize(features.float(),dim=-1) @ F.normalize(query,dim=-1).T
        if not torch.isfinite(scores).all(): raise ValueError('Điểm tương đồng không hữu hạn')
        return scores.squeeze(1)
