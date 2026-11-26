import struct
import os
import sys
import re
import json

class TXDTool:
    def __init__(self, encoding='euc_jis_2004'):
        self.encoding = encoding

    def load_translation_table(self):
        table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XENOSAGA KOR-JPN.json')
        with open(table_path, 'r', encoding='utf-8-sig') as f:
            table = json.load(f)['replace-table']
        if not isinstance(table, dict) or any(
            not isinstance(key, str) or len(key) != 1
            or not isinstance(value, str) or not value
            for key, value in table.items()
        ):
            raise ValueError('JSON의 replace-table은 한 글자 키와 비어 있지 않은 문자열 값이어야 합니다.')
        return str.maketrans(table)

    def get_data_start(self, data):
        # In these TXDs the string pool starts at the lowest nonzero pointer.
        start = len(data)
        for offset in range(0, len(data) - 3, 4):
            if offset >= start:
                break
            ptr, = struct.unpack_from('<I', data, offset)
            if ptr:
                if not offset + 4 <= ptr < len(data):
                    raise ValueError('TXD 문자열 포인터가 범위를 벗어났습니다.')
                start = min(start, ptr)
        if start == len(data) or start % 4:
            raise ValueError('TXD 포인터 테이블의 경계를 확인할 수 없습니다.')
        return start

    def safe_decode(self, raw_bytes):
        """디코딩 에러 발생 시 2바이트씩 묶어서 [0xXXXX] 형태로 추출"""
        decoded_str = ""
        curr_bytes = raw_bytes
        
        while curr_bytes:
            try:
                # 현재 남은 바이트를 통째로 디코딩 시도
                decoded_str += curr_bytes.decode(self.encoding).replace('\r\n', '\n')
                break
            except UnicodeDecodeError as e:
                # 에러 발생 전까지의 정상적인 부분은 추가
                decoded_str += curr_bytes[:e.start].decode(self.encoding).replace('\r\n', '\n')
                
                # 에러 지점부터 2바이트를 추출 (남은 바이트가 1바이트뿐이면 1바이트만)
                step = 2 if len(curr_bytes) >= e.start + 2 else 1
                bad_unit = curr_bytes[e.start : e.start + step]
                
                decoded_str += f"[0x{bad_unit.hex().upper()}]"
                curr_bytes = curr_bytes[e.start + step:]
        return decoded_str

    def extract(self, txd_path):
        if not os.path.exists(txd_path):
            print(f"오류: {txd_path} 파일을 찾을 수 없습니다.")
            return

        with open(txd_path, 'rb') as f:
            data = f.read()

        data_start = self.get_data_start(data)
        ptr_count = data_start // 4
        
        pointers = [struct.unpack('<I', data[i*4:i*4+4])[0] for i in range(ptr_count)]
        output_txt = txd_path + ".txt"

        with open(output_txt, 'w', encoding='utf-8') as f:
            for i, ptr in enumerate(pointers):
                if ptr == 0 or ptr >= len(data):
                    f.write(f"[{i}]\n")
                    continue
                
                end = data.find(b'\x00', ptr)
                raw = data[ptr:end] if end != -1 else data[ptr:]
                
                text = self.safe_decode(raw)
                f.write(f"[{i}]{text}\n")
        
        print(f"▶ 추출 완료: {output_txt}")

    def repack(self, original_txd, modified_txt):
        if not os.path.exists(original_txd) or not os.path.exists(modified_txt):
            print("오류: 파일이 없습니다.")
            return

        translation_table = self.load_translation_table()

        with open(modified_txt, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        with open(original_txd, 'rb') as f:
            orig_data = f.read()
        
        data_start = self.get_data_start(orig_data)
        ptr_count = data_start // 4
        original_ptrs = struct.unpack_from(f'<{ptr_count}I', orig_data)
        markers = list(re.finditer(r'^\[([0-9]+)\]', content, re.M))
        if [m[1] for m in markers] != [str(i) for i in range(ptr_count)] or content[:markers[0].start()].strip():
            raise ValueError(f'TXT 항목 번호는 [0]부터 [{ptr_count - 1}]까지 순서대로 있어야 합니다.')
        ends = [m.start() for m in markers[1:]] + [len(content)]
        new_texts = [content[m.end():end].rstrip('\n') for m, end in zip(markers, ends)]

        new_ptr_table = []
        new_data_pool = b""
        current_offset = data_start 

        for i, text in enumerate(new_texts):
            if "[menu_sita]" in text:
                text = text.replace("[menu_sita]", "").replace(" ", "、")

            raw_pool = b""
            tokens = re.split(r'(\[0x[0-9A-Fa-f]+\])', text)
            for token in tokens:
                hex_match = re.fullmatch(r'\[0x([0-9A-Fa-f]+)\]', token)
                if hex_match:
                    hex_str = hex_match.group(1)
                    if len(hex_str) % 2 != 0: hex_str = "0" + hex_str
                    raw_pool += bytes.fromhex(hex_str)
                else:
                    token = token.translate(translation_table)
                    unmapped = sorted(set(re.findall(r'[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]', token)))
                    if unmapped:
                        raise ValueError(f'[{i}] JSON 대응표에 없는 한글: {" ".join(unmapped)}')
                    try:
                        raw_pool += token.replace('\n', '\r\n').encode(self.encoding)
                    except UnicodeEncodeError as e:
                        raise ValueError(f'[{i}] 인코딩할 수 없는 문자: {e.object[e.start:e.end]!r}') from e

            if b'\0' in raw_pool:
                raise ValueError(f'[{i}] 문자열에 NUL을 넣을 수 없습니다.')
            if not raw_pool:
                new_ptr_table.append(None if original_ptrs[i] else 0)
                continue
            new_ptr_table.append(current_offset)
            new_data_pool += raw_pool + b'\x00'
            current_offset += len(raw_pool) + 1

        # A live empty string must point to a terminator, not to the table.
        if None in new_ptr_table:
            new_data_pool += b'\0'
            new_ptr_table = [current_offset if ptr is None else ptr for ptr in new_ptr_table]

        output_path = os.path.splitext(original_txd)[0] + '_repacked.txd'
        with open(output_path, 'wb') as f:
            for ptr in new_ptr_table:
                f.write(struct.pack('<I', ptr))
            f.write(new_data_pool)
            
        print(f"▶ 리패킹 완료: {output_path}")

if __name__ == "__main__":
    tool = TXDTool()
    if len(sys.argv) == 2 and sys.argv[1].lower() == 'r':
        try:
            name = input('리빌드할 파일명 (Location / Tutorial): ').strip().strip('"').lower()
        except (EOFError, KeyboardInterrupt):
            print('\n리빌드를 취소했습니다.')
            sys.exit(1)
        targets = {stem.lower() + suffix: stem for stem in ('Location', 'Tutorial')
                   for suffix in ('', '.txd', '.txd.txt')}
        if name not in targets:
            print('Location 또는 Tutorial을 입력해 주세요. 파일은 변경하지 않았습니다.')
            sys.exit(1)
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), targets[name] + '.txd')
        if not os.path.isfile(source) or not os.path.isfile(source + '.txt'):
            print('원본 TXD 또는 번역 TXT가 없습니다. 파일은 변경하지 않았습니다.')
            sys.exit(1)
        sys.argv.append(source)
    if len(sys.argv) < 3:
        print("사용법:")
        print("  추출: python txd_tool.py e 파일명.txd")
        print("  파일명 입력 리팩: python txd_tool.py r")
        print("  리팩: python txd_tool.py r 파일명.txd 파일명.txt")
        print("  리팩 시 스크립트 옆 XENOSAGA KOR-JPN.json으로 한글을 자동 변환합니다.")
    else:
        mode = sys.argv[1].lower()
        if mode == 'e': tool.extract(sys.argv[2])
        elif mode == 'r':
            try:
                tool.repack(sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else sys.argv[2] + ".txt"))
            except (OSError, ValueError, KeyError) as e:
                print(f"오류: {e}")
                sys.exit(1)
