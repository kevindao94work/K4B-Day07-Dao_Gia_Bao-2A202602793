# CP2 — Data collection record

**Corpus:** data/tiki-doi-tra/  
**Topic:** Chính sách đổi trả, bảo hành và quy định nhà bán trên Tiki Việt Nam  
**Retrieved:** 2026-09-20

## Source manifest

data/urls.csv và data/tiki-doi-tra/sources.csv ghi tám URL Tiki, ánh xạ 1-1 với tám Markdown đã làm sạch. Mỗi file có frontmatter bắt buộc: doc_id, title, source_url, retrieved_at, document_version, audience, cùng category, language, platform, market, channel và source_scope.

| Audience | Số tài liệu | Nội dung |
|---|---:|---|
| buyer | 2 | Quyền đổi trả 30 ngày và chương trình 365 ngày |
| seller | 6 | Dropship, FBT, NGON, SD, FAQ và RMA |

## Làm sạch và kiểm chứng

Chỉ giữ nội dung điều khoản, trách nhiệm, mốc thời gian, điều kiện và quy trình; menu, banner, chân trang và nội dung điều hướng đã bị loại trước chunking. document_version là not-stated khi nguồn không công bố số hiệu/phiên bản.

Đã chạy python scripts/check_cp2.py. Kết quả: 8 file hợp lệ, manifest 1-1, audience gồm buyer và seller, và cả năm câu benchmark có evidence trong nguồn.

## Giới hạn nguồn

Bảng Shopee được cung cấp trong yêu cầu không được thêm vào corpus vì thiếu URL/trích đoạn nguồn kiểm chứng và điều khoản Shopee yêu cầu chấp thuận bằng văn bản trước khi thu thập hoặc sao chép nội dung. Corpus hiện chỉ dùng nguồn Tiki có provenance được lưu đầy đủ.
