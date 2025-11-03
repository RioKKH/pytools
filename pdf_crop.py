#!/usr/bin/env python


from PyPDF2 import PdfReader, PdfWriter

# reader = PdfReader("temp_even_only.pdf")
reader_out_1_odd = PdfReader("part1_odd.pdf")
reader_out_1_even = PdfReader("part1_even.pdf")
reader_out_2_odd = PdfReader("part2_odd.pdf")
reader_out_2_even = PdfReader("part2_even.pdf")

writer_out_1_odd_cropped = PdfWriter()
writer_out_1_even_cropped = PdfWriter()
writer_out_2_odd_cropped = PdfWriter()
writer_out_2_even_cropped = PdfWriter()
writer = PdfWriter()

for page in reader_out_1_odd.pages:
    mediabox = page.mediabox
    page.mediabox.lower_left = (0, float(mediabox.height) * 0.15)
    writer_out_1_odd_cropped.add_page(page)
with open("math_part1_odd_cropped.pdf", "wb") as f:
    writer_out_1_odd_cropped.write(f)

for page in reader_out_1_even.pages:
    mediabox = page.mediabox
    print(f"width {mediabox.width}, height {mediabox.height}")
    print(mediabox.upper_right)
    page.mediabox.upper_right = (mediabox.width, float(mediabox.height) * 0.85)
    # page.mediabox.upper_right = (float(mediabox.width), float(mediabox.height) * 0.85)
    print(mediabox.upper_right)
    writer_out_1_even_cropped.add_page(page)
with open("math_part1_even_cropped.pdf", "wb") as f:
    writer_out_1_even_cropped.write(f)

for page in reader_out_2_odd.pages:
    mediabox = page.mediabox
    print(f"width {mediabox.width}, height {mediabox.height}")
    page.mediabox.upper_right = (float(mediabox.width), float(mediabox.height) * 0.85)
    writer_out_2_odd_cropped.add_page(page)
with open("math_part2_odd_cropped.pdf", "wb") as f:
    writer_out_2_odd_cropped.write(f)

for page in reader_out_2_even.pages:
    mediabox = page.mediabox
    page.mediabox.lower_left = (0, float(mediabox.height) * 0.15)
    writer_out_2_even_cropped.add_page(page)
with open("math_part2_even_cropped.pdf", "wb") as f:
    writer_out_2_even_cropped.write(f)


# 1つ目のファイルを組み合わせる
# pdftk A=math_part1_odd_cropped.pdf B=math_part1_even_cropped.pdf shuffle A B output math_part1_final.pdf

# 2つ目のファイルを組み合わせる
# pdftk A=math_part2_odd_cropped.pdf B=math_part2_even_cropped.pdf shuffle A B output math_part2_final.pdf

# 2つのpdfを結合する
# pdftk math_part1_final.pdf math_part2_final.pdf cat output math_complete.pdf
