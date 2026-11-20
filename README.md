# LinCIR × GeneCIS — demo truy hồi ảnh

Ứng dụng Gradio chạy checkpoint LinCIR ViT-L của tác giả trên hai tác vụ GeneCIS **focus_object** và **change_object**. Mỗi truy vấn dùng 1 ảnh tham chiếu, điều kiện gốc và gallery chính thức 15 ảnh (1 target + 14 distractor).

## Hai tác vụ GeneCIS trong demo

Đây là hai tác vụ đánh giá do GeneCIS định nghĩa, không phải nhãn do ứng dụng tự tạo:

- **Focus Object**: condition chọn một đối tượng có trong ảnh tham chiếu làm trọng tâm; tìm ảnh có đối tượng đó và bối cảnh tương tự.
- **Change Object**: condition chỉ định một đối tượng cần được thêm vào bối cảnh; tìm ảnh có bối cảnh tương tự ảnh tham chiếu nhưng có thêm đối tượng đó. Đây là truy xuất ảnh có sẵn, không phải sinh hoặc chỉnh sửa pixel.

Nguồn: [GeneCIS, CVPR 2023, mục 4 và Hình 2](https://sgvaze.github.io/genecis/resources/paper.pdf). Mỗi tác vụ object có 1.960 truy vấn và 15 ảnh ứng viên cho mỗi truy vấn (Bảng 1 của GeneCIS).

## Nguồn annotation và cách đóng gói

Hai file nén trong `data/annotations` chứa nguyên nội dung JSON của GeneCIS tại commit `0c5c968c01b8ccffcc0990af2b430d21d7a12f49`:

| File trong demo | File gốc GeneCIS | Git blob SHA-1 của nội dung sau giải nén |
| --- | --- | --- |
| `focus_object.json.gz` | [genecis/focus_object.json](https://github.com/facebookresearch/genecis/blob/0c5c968c01b8ccffcc0990af2b430d21d7a12f49/genecis/focus_object.json) | `8aa323433948d115beb48d18058208dff586edd0` |
| `change_object.json.gz` | [genecis/change_object.json](https://github.com/facebookresearch/genecis/blob/0c5c968c01b8ccffcc0990af2b430d21d7a12f49/genecis/change_object.json) | `f06e61063c121f1ec68bea021ab3b08887d517d5` |

Ứng dụng không tự tạo nhãn condition, target hay distractor. Code chỉ đọc annotation, chọn tập con có seed, tải ảnh COCO theo ID và tạo/cache embedding. Đuôi `.gz` là cách đóng gói để giảm dung lượng; không phải bước sinh dữ liệu mới.

## Chạy trên macOS / Linux

Cần Python 3.10–3.12, `make`, Internet lần đầu và đủ dung lượng cho checkpoint CLIP, LinCIR và ảnh. Apple M1 tự chọn MPS; máy NVIDIA tự chọn CUDA; có thể chọn CPU trên giao diện. Với CUDA, trước tiên cài PyTorch phù hợp cấu hình máy nếu cần.

```bash
make setup
make run
```

Mở http://127.0.0.1:7860. Lần đầu không cần chuẩn bị dataset thủ công:

1. Chọn Focus Object hoặc Change Object; chọn 20, 100 hoặc toàn bộ 1.960 truy vấn; chọn thiết bị.
2. Nhấn **Tải và chuẩn bị**. Giao diện hiển thị tiến trình kiểm tra/tải ảnh COCO, nạp checkpoint, mã hóa gallery. CLIP và checkpoint được lưu trong cache Hugging Face mặc định. Quá trình tải model có thể mất vài phút; trạng thái thể hiện theo giai đoạn, không phải phần trăm tải model.
3. Khi trạng thái **Sẵn sàng**, chọn truy vấn, nhấn **Tìm ảnh**. Ứng dụng đánh dấu target theo annotation và hiển thị thứ hạng.
4. Khi đổi tác vụ hoặc số truy vấn, chuẩn bị lại. Ảnh đã tải được tái sử dụng; embedding lưu trong `data/embeddings`. Nếu mạng lỗi, bấm chuẩn bị lại.

20/100 truy vấn được lấy ngẫu nhiên có seed 42 từ annotation; 20 là tập con của 100. Số truy vấn thay đổi, nhưng 15 ứng viên trong mỗi truy vấn được giữ nguyên. 1.960 là toàn bộ truy vấn cho **một** trong hai tác vụ, không phải toàn bộ bốn tác vụ GeneCIS. Annotation gốc được đóng gói từ commit `0c5c968c01b8ccffcc0990af2b430d21d7a12f49`; COCO ảnh tải trực tiếp từ máy chủ COCO. Không đưa ảnh/embedding/model vào Git.

Kết quả một truy vấn không phải Recall@K toàn tập. Demo dùng prompt `a photo of $ that {condition}` và pseudo-token LinCIR; trình bày là demo, không khẳng định tái lập điểm trong paper nếu chưa chạy đánh giá đầy đủ và kiểm tra giao thức. Truy vấn chính thức có ảnh tham chiếu/điều kiện cố định. Mục truy vấn tự do hỗ trợ tải ảnh và nhập câu tiếng Anh trên gallery chung của các ảnh đã chuẩn bị; đây không phải giao thức benchmark GeneCIS.

## Kiểm thử

```bash
make check
```

Kiểm thử offline xác nhận cấu trúc gallery và chọn mẫu. Không tải dữ liệu/model. Lần chạy đầy đủ phụ thuộc truy cập mạng và khả năng tương thích thiết bị.

## Nguồn và giấy phép

- LinCIR: https://github.com/navervision/lincir — CC BY-NC 4.0, checkpoint của tác giả; kiến trúc Phi theo code chính thức.
- GeneCIS: https://github.com/facebookresearch/genecis — annotation CC BY-NC 4.0 (bản giấy phép trong `data/annotations`).
- Ảnh COCO: https://cocodataset.org/#termsofuse — bản quyền ảnh thuộc từng chủ sở hữu; kiểm tra điều khoản trước khi tái phân phối.
