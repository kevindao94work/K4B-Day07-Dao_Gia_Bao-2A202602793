# Câu trả lời dựa trên top-3 của fixed

Các câu trả lời dưới đây được tổng hợp có dẫn nguồn từ top-3 trong benchmark.txt. Repo chỉ cấu hình demo_llm để xem trước prompt, nên đây là bản trả lời grounded do agent đánh giá từ ngữ cảnh truy xuất, không phải output của một LLM sinh văn bản đã tích hợp. Khi top-3 thiếu evidence, agent từ chối suy đoán.

## Q1 — Chương trình 365 ngày

**Trả lời:** Chương trình áp dụng cho ngành **Điện gia dụng** và **Thiết bị số** thuộc nhà bán **Tiki Trading**. [1]

**Chunk/source:** [1] tiki-doi-tra-365#0 — https://tiki.vn/blog/chuong-trinh-doi-tra-365/ (hạng 1, score 0.743338).  
**Bằng chứng:** Đủ; cùng chunk nêu cả hai ngành hàng và nhà bán. Bộ lọc audience=buyer đã dùng.

## Q2 — Bằng chứng của Dropship

**Trả lời:** Khi từ chối đổi trả hoặc bảo hành, nhà bán có thể nộp biên bản bàn giao/đồng kiểm, ảnh hoặc video đóng gói/khui mở, hoặc biên bản thẩm định của hãng. [1]

**Chunk/source:** [1] tiki-dropship-doi-tra-bao-hanh#0 — https://hocvien.tiki.vn/faq/huong-dan-quy-trinh-xu-ly-doi-tra-bao-hanh-mo-hinh-dropship/ (hạng 1, score 0.572443).  
**Bằng chứng:** Đủ; ba loại bằng chứng được liệt kê trực tiếp.

## Q3 — Hàng hoàn NGON hư hỏng do vận chuyển

**Trả lời:** Trong vòng **24 giờ** từ khi Tiki trả hàng, nhà bán phải liên hệ Tiki, xác nhận thông tin và giá trị bồi thường, rồi cung cấp hồ sơ/chứng từ qua Seller Center. [2]

**Chunk/source:** [2] tiki-ngon-doi-tra-boi-thuong#1 — https://hocvien.tiki.vn/faq/mo-hinh-ngon-huong-dan-quy-trinh-xu-ly-doi-tra-boi-thuong/ (hạng 2, score 0.501842).  
**Bằng chứng:** Đủ; chunk hạng 2 chứa thời hạn, điều kiện hư hỏng do vận chuyển và các bước cần làm.

## Q4 — Hạn rút hàng FBT

**Trả lời:** Không tìm thấy thời hạn rút hàng FBT trong ba chunk đã truy xuất; chưa đủ căn cứ để trả lời. 

**Chunk/source:** Top-3 là tiki-faq-doi-tra-bao-hanh#1, tiki-faq-doi-tra-bao-hanh#0 và tiki-doi-tra-365#0. Không chunk nào nêu thời hạn rút hàng FBT.  
**Bằng chứng:** Thiếu trong ngữ cảnh agent. Tài liệu gốc có câu trả lời ở tiki-fbt-doi-tra-bao-hanh#0, nhưng chunk đó không được truy xuất; không dùng gold answer để lấp chỗ thiếu.

## Q5 — Đổi trả miễn phí

**Trả lời:** Trang cam kết đổi trả của Tiki nêu **30 ngày đổi trả miễn phí**; trang cũng nêu cam kết hoàn **200%** nếu hàng giả. [2] Có một chương trình 365 ngày riêng cho sản phẩm Điện gia dụng/Thiết bị số của Tiki Trading khi có lỗi kỹ thuật do nhà sản xuất. [1]

**Chunk/source:** [2] tiki-doi-tra-30-ngay#0 — https://tiki.vn/thong-tin/tiki-doi-tra-de-dang-an-tam-mua-sam (hạng 2, score 0.572190); [1] tiki-doi-tra-365#0 — https://tiki.vn/blog/chuong-trinh-doi-tra-365/ (hạng 1, score 0.594412).  
**Bằng chứng:** Đủ cho mốc 30 ngày và thông tin 200%; nguồn 30 ngày không nêu chi tiết điều kiện hay ngoại lệ nên không suy diễn thêm. A/B với audience=buyer giữ nguyên hai vị trí đầu; vị trí thứ ba đổi từ seller FAQ sang buyer 365#1.
