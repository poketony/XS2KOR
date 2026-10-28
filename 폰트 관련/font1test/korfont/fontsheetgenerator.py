"""Render the approved font1 Korean cells; preserve all other atlas cells."""
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_SHA256 = 'c89bc43027dc7cde5726e96223376f8eec09302b2fc1f8147fd5b57cfc376118'
SOURCES = (('green', 2, 1024), ('blue', 3, 1024), ('alpha', 4, 302))


def read_lists(folder):
    result = []
    for suffix, sheet, count in SOURCES:
        text = (folder / f'hangul_2048_final_{suffix}.txt').read_text(encoding='utf-8-sig').strip()
        if len(text) != count or any(not '\uac00' <= char <= '\ud7a3' for char in text):
            raise ValueError(f'{suffix}: expected exactly {count} Korean syllables')
        result.append((sheet, text))
    if len(set(''.join(text for _, text in result))) != 2350:
        raise ValueError('Korean lists must contain 2350 unique syllables')
    return result


def generate(folder, sheets_folder, font_path=None):
    folder, sheets_folder = Path(folder), Path(sheets_folder)
    font_path = folder / 'Pretendard-SemiBold.otf' if font_path is None else Path(font_path)
    if hashlib.sha256(font_path.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('Pretendard-SemiBold.otf differs from the approved source')
    font = ImageFont.truetype(str(font_path), 20)
    if font.getname() != ('Pretendard', 'SemiBold'):
        raise ValueError('Unexpected source font face')
    results = {}
    for number, chars in read_lists(folder):
        path = sheets_folder / f'font1_sheet{number}.png'
        with Image.open(path) as source:
            if source.size != (640, 768) or source.mode != 'L':
                raise ValueError(f'{path.name}: expected 640x768 grayscale sheet')
            sheet = source.copy()
        for index, char in enumerate(chars):
            # Check the padded glyph before cropping: the engine draws only 19 columns.
            padded = Image.new('L', (60, 64))
            ImageDraw.Draw(padded).text((21, 39), char, font=font, anchor='ls', fill=255)
            bounds = padded.getbbox()
            if bounds is None:
                raise ValueError(f'{char}: empty glyph')
            left, top, right, bottom = (value - 20 for value in bounds)
            if left < 0 or top < 0 or right > 19 or bottom > 24:
                raise ValueError(f'{char}: glyph exceeds visible cell; no output saved')
            sheet.paste(padded.crop((20, 20, 40, 44)), ((index % 32) * 20, (index // 32) * 24))
        results[path] = sheet
    return results


def main():
    folder = Path(__file__).resolve().parent
    # Validate the complete batch before replacing the editable sheets.
    results = generate(folder, folder.parent)
    for path, image in results.items():
        image.save(path)
        print(f'[SAVED] {path.name}')
    print('[OK] 2350 Korean glyphs; Pretendard SemiBold 20px; baseline 19; no stretching.')
    print('Run font1_rebuild.bat in the parent folder to create font1.tex.new.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f'[ERROR] {exc}')
