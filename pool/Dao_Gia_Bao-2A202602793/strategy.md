# Chiến lược cá nhân

- **Thành viên:** Đào Gia Bảo — 2A202602793
- **Strategy:** fixed (FixedSizeChunker)
- **Tham số:** chunk_size = 800 ký tự; overlap = 50 ký tự; top_k = 3.
- **Corpus:** 8 tài liệu Tiki trong data/tiki-doi-tra/, tạo 19 chunk.
- **Embedding:** local sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, vector chuẩn hóa 384 chiều.
- **Metadata filter:** Q1 và Q5 dùng audience=buyer; Q2–Q4 không lọc. Q5 được chạy thêm không filter để đối chiếu A/B.

Tôi chọn fixed để có baseline đơn giản, cùng kích thước trần 800 ký tự với các chiến lược khác. Overlap 50 ký tự giảm nguy cơ cắt rời thông tin ở ranh giới chunk. Điểm yếu đo được là câu FBT Q4: chunk chứa mốc 32 ngày không lọt top-3 dù đã có trong corpus.

Lệnh chạy:

```bash
.venv/bin/python bench.py --provider local --strategy fixed --output pool/Dao_Gia_Bao-2A202602793/benchmark.txt
```

Lượt chạy này đọc 26 vector local đã cache và không có cache miss. Venv chưa cài bộ sinh embedding; cache được tạo trước đó bằng môi trường hệ thống có sentence-transformers 5.6.0. Tôi đã nhúng lại Q1 bằng model thật trên môi trường đó và so với vector cache: 384 chiều, sai khác lớn nhất bằng 0.0. Nếu corpus hoặc câu hỏi đổi, cần cài model trong venv hoặc chạy bằng môi trường hệ thống để tạo vector mới; chương trình không tự chuyển sang mock.

Không sửa riêng file nào trong src/ cho submission này; không cần changed_files.patch.
