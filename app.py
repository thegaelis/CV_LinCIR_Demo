"""Local research demo: GeneCIS object tasks, LinCIR ViT-L."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, os, urllib.request
import numpy as np
from PIL import Image, ImageOps
import torch
import gradio as gr
from core import ROOT, TASKS, load_cases, required_ids, case_ids, candidates, image_path
from model import LinCIR, MODEL, REVISION

STATE={'ready':False,'model':None,'cases':{},'features':{},'task':None,'limit':None}

def read_image(path):
    with Image.open(path) as image: return ImageOps.exif_transpose(image).convert('RGB')

def download_image(image_id):
    path=image_path(image_id)
    if path.exists():
        try:
            with Image.open(path) as im: im.verify()
            return
        except (OSError,ValueError): path.unlink()
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    url=f'http://images.cocodataset.org/val2017/{image_id:012d}.jpg'
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'LinCIR-education-demo/1.0'})
        with urllib.request.urlopen(request,timeout=40) as response, temp.open('wb') as out:
            while block:=response.read(1024*256): out.write(block)
        with Image.open(temp) as image: image.verify()
        temp.replace(path)
    finally: temp.unlink(missing_ok=True)

def prepare(task,limit,device,progress=gr.Progress()):
    STATE.update(ready=False,features={},cases={})
    try:
        limit=int(limit); cases=load_cases(task,limit); ids=required_ids(cases)
        yield f'1/3 · Cần {len(ids)} ảnh cho {limit} truy vấn. Kiểm tra và tải từ COCO…',gr.update(choices=[],value=None),gr.update(interactive=False)
        errors=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures={pool.submit(download_image,i):i for i in ids}
            for done,future in enumerate(as_completed(futures),1):
                try: future.result()
                except Exception as e: errors.append((futures[future],str(e)))
                if done%20==0 or done==len(ids):
                    progress((done,len(ids)),desc='1/3 · Kiểm tra / tải ảnh COCO')
                    yield f'1/3 · Đã kiểm tra {done}/{len(ids)} ảnh; lỗi: {len(errors)}.',gr.update(),gr.update(interactive=False)
        if errors: raise RuntimeError(f'Không tải được {len(errors)} ảnh (ví dụ {errors[:3]}). Bấm Chuẩn bị để thử lại; ảnh đã tải được giữ lại.')
        yield '2/3 · Đang tải/nạp CLIP và checkpoint LinCIR. Lần đầu có thể mất vài phút…',gr.update(),gr.update(interactive=False)
        if STATE['model'] is None or STATE.get('device_requested')!=device:
            STATE['model']=None
            STATE['model']=LinCIR(device)
            STATE['device_requested']=device
        model=STATE['model']
        signature=hashlib.sha256(json.dumps([MODEL,REVISION,str(model.device),str(model.dtype),[(i,image_path(i).stat().st_size,image_path(i).stat().st_mtime_ns) for i in ids]]).encode()).hexdigest()
        cache_dir=ROOT/'data'/'embeddings'; cache_dir.mkdir(parents=True,exist_ok=True)
        cache_file=cache_dir/f'{signature}.npz'
        embeddings=None
        if cache_file.exists():
            try:
                with np.load(cache_file,allow_pickle=False) as saved:
                    if saved['ids'].tolist()==ids: embeddings=saved['features'].copy()
            except (ValueError,KeyError,OSError): pass
        if embeddings is None:
            batches=[]
            for start in range(0,len(ids),8):
                batch=ids[start:start+8]
                features=model.encode_images([read_image(image_path(i)) for i in batch])
                batches.append(features.numpy())
                done=min(start+len(batch),len(ids))
                progress((done,len(ids)),desc='3/3 · Mã hóa ảnh')
                if start%40==0 or done==len(ids):
                    yield f'3/3 · Đã mã hóa {done}/{len(ids)} ảnh…',gr.update(),gr.update(interactive=False)
            embeddings=np.concatenate(batches)
            if not np.isfinite(embeddings).all(): raise RuntimeError('Embedding không hữu hạn.')
            temp=cache_file.with_suffix('.tmp.npz')
            np.savez_compressed(temp,ids=np.asarray(ids,dtype=np.int64),features=embeddings)
            os.replace(temp,cache_file)
        STATE.update(ready=True,cases=dict(cases),features={i:torch.from_numpy(f.copy()) for i,f in zip(ids,embeddings)},task=task,limit=limit)
        choices=[(f"Truy vấn #{i} · {case['condition']}",str(i)) for i,case in cases]
        yield f'Sẵn sàng · {task} · {limit} truy vấn · {len(ids)} ảnh nguồn · {model.device}. Mỗi truy vấn dùng đúng 15 ảnh ứng viên.',gr.update(choices=choices,value=choices[0][1]),gr.update(interactive=True)
    except Exception as e:
        STATE['ready']=False
        yield f'Chuẩn bị chưa hoàn tất: {type(e).__name__}: {e}',gr.update(choices=[],value=None),gr.update(interactive=False)

def show_case(selected):
    if not STATE['ready'] or selected is None: return None,''
    case=STATE['cases'][int(selected)]; ref,_,gallery=case_ids(case)
    return read_image(image_path(ref)),f"Điều kiện chính thức: **{case['condition']}** · {len(gallery)+1} ảnh ứng viên"

def run_query(task,limit,selected,k):
    if not STATE['ready'] or STATE['task']!=task or STATE['limit']!=int(limit): raise gr.Error('Cấu hình đã đổi. Hãy chuẩn bị lại trước khi truy vấn.')
    if selected is None or int(selected) not in STATE['cases']: raise gr.Error('Hãy chọn truy vấn đã chuẩn bị.')
    case=STATE['cases'][int(selected)]; ref,target,_=case_ids(case); choices=candidates(case,seed=42)
    try: features=torch.stack([STATE['features'][i] for i in choices])
    except KeyError as e: raise gr.Error(f'Thiếu embedding {e}; hãy chuẩn bị lại.')
    scores=STATE['model'].rank(STATE['features'][ref],case['condition'],features)
    positions=torch.argsort(scores,descending=True).tolist(); k=min(int(k),len(positions))
    images=[(read_image(image_path(choices[j])),f"#{rank} · {scores[j]:.3f}" + (' · Đáp án gán nhãn' if choices[j]==target else '')) for rank,j in enumerate(positions[:k],1)]
    target_rank=next((rank for rank,j in enumerate(positions,1) if choices[j]==target),None)
    return images,f'Ảnh đích theo annotation xếp **#{target_rank}/15**. Điểm cosine không phải xác suất. Đây là kết quả một truy vấn, không phải Recall@K toàn bộ benchmark.'

def free_query(task,limit,reference,condition,k):
    if not STATE['ready'] or STATE['task']!=task or STATE['limit']!=int(limit):
        raise gr.Error('Hãy chuẩn bị dữ liệu cho cấu hình hiện tại trước.')
    if reference is None or not condition or not condition.strip():
        raise gr.Error('Hãy chọn ảnh tham chiếu và nhập yêu cầu tiếng Anh.')
    if '$' in condition: raise gr.Error('Câu yêu cầu không được chứa ký hiệu $.')
    image=ImageOps.exif_transpose(reference).convert('RGB')
    model=STATE['model']; ref=model.encode_images([image])[0]
    ids=sorted(STATE['features']); features=torch.stack([STATE['features'][i] for i in ids])
    scores=model.rank(ref,condition.strip(),features)
    positions=torch.argsort(scores,descending=True).tolist()
    fingerprint=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    result=[]
    for j in positions:
        candidate=read_image(image_path(ids[j]))
        digest=hashlib.sha256(str(candidate.size).encode()+candidate.tobytes()).hexdigest()
        if digest==fingerprint: continue
        result.append((candidate,f'#{len(result)+1} · {scores[j]:.3f}'))
        if len(result)>=int(k): break
    return result,f'Tìm trong {len(ids)} ảnh nguồn đã chuẩn bị. Đây là gallery chung cho demo, không phải giao thức 15 ảnh của GeneCIS.'

with gr.Blocks(title='LinCIR · GeneCIS',theme=gr.themes.Soft(primary_hue='indigo')) as app:
    gr.Markdown('# LinCIR × GeneCIS\nDemo zero-shot composed retrieval · **Focus Object** và **Change Object** (COCO validation).')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 1 · Cấu hình và chuẩn bị')
            task=gr.Dropdown(choices=[('Focus Object','focus_object'),('Change Object','change_object')],value='focus_object',label='Tác vụ GeneCIS')
            limit=gr.Dropdown(choices=[('20 truy vấn · thử nhanh',20),('100 truy vấn',100),('Toàn bộ · 1.960 truy vấn',1960)],value=20,label='Phạm vi chuẩn bị')
            device=gr.Dropdown(choices=['auto','cuda','mps','cpu'],value='auto',label='Thiết bị')
            prep=gr.Button('Tải và chuẩn bị',variant='primary')
            status=gr.Markdown('Chưa chuẩn bị. Nhấn nút để tải ảnh, nạp mô hình và mã hóa dữ liệu.')
            gr.Markdown('Lần đầu cần Internet và dung lượng để tải checkpoint, CLIP và ảnh. Tác vụ mới hoặc số truy vấn thay đổi cần chuẩn bị lại.')
        with gr.Column(scale=2):
            gr.Markdown('### 2 · Truy vấn chính thức')
            selected=gr.Dropdown(choices=[],label='Chọn truy vấn')
            with gr.Row():
                reference=gr.Image(type='pil',label='Ảnh tham chiếu',interactive=False)
                description=gr.Markdown('Chuẩn bị dữ liệu để xem điều kiện.')
            k=gr.Slider(1,15,value=5,step=1,label='Top-K')
            run=gr.Button('Tìm ảnh',interactive=False)
            note=gr.Markdown('')
            output=gr.Gallery(label='Kết quả',columns=5,object_fit='contain',height=400)
    with gr.Accordion('Truy vấn tự do với gallery đã chuẩn bị',open=False):
        gr.Markdown('Chọn ảnh và nhập yêu cầu tiếng Anh. Kết quả tìm trên toàn bộ ảnh nguồn đã chuẩn bị; không có nhãn đáp án GeneCIS.')
        with gr.Row():
            custom_ref=gr.Image(type='pil',label='Ảnh tham chiếu')
            with gr.Column():
                custom_text=gr.Textbox(label='Yêu cầu',placeholder='is blue instead of red')
                custom_k=gr.Slider(1,20,value=5,step=1,label='Top-K')
                custom_button=gr.Button('Tìm trong gallery')
        custom_note=gr.Markdown('')
        custom_output=gr.Gallery(label='Kết quả truy vấn tự do',columns=5,object_fit='contain',height=400)
        custom_button.click(free_query,[task,limit,custom_ref,custom_text,custom_k],[custom_output,custom_note])
    prep.click(prepare,[task,limit,device],[status,selected,run])
    selected.change(show_case,[selected],[reference,description])
    run.click(run_query,[task,limit,selected,k],[output,note])
    gr.Markdown('Theo giao thức GeneCIS, mỗi truy vấn gồm target và 14 ảnh nhiễu. Annotation: facebookresearch/genecis; ảnh: COCO 2017 val. Kết quả phụ thuộc checkpoint, phần cứng và dữ liệu.')

if __name__=='__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7860)
