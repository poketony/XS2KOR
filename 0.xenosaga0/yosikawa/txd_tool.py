import struct
import os
import sys
import re

class TXDTool:
    def __init__(self, encoding='euc-jp'):
        self.encoding = encoding

    def get_data_start(self, data):
        """4바이트 묶음 중 상위 2바이트가 0이 아닌 지점이 데이터의 시작"""
        for i in range(0, len(data), 4):
            if i + 4 > len(data): break
            upper_half = struct.unpack('<H', data[i+2:i+4])[0]
            if upper_half != 0:
                return i
        return len(data)

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

        with open(modified_txt, 'r', encoding='utf-8') as f:
            content = f.read()
            parts = re.split(r'\[\d+\]', content)
            if parts:
                new_texts = [p.rstrip('\n') for p in parts[1:]]

        with open(original_txd, 'rb') as f:
            orig_data = f.read()
        
        data_start = self.get_data_start(orig_data)
        ptr_count = data_start // 4

        new_ptr_table = []
        new_data_pool = b""
        current_offset = data_start 

        for i in range(ptr_count):
            if i < len(new_texts):
                text = new_texts[i]
                if not text:
                    new_ptr_table.append(0)
                    continue

                # --- 추가된 기능: [menu_sita] 태그 처리 ---
                if "[menu_sita]" in text:
                    # 태그 제거 및 공백을 '、'로 변경
                    text = text.replace("[menu_sita]", "").replace(" ", "、")
                # ----------------------------------------
                
                raw_pool = b""
                tokens = re.split(r'(\[0x[0-9A-Fa-f]+\])', text)
                for token in tokens:
                    hex_match = re.match(r'\[0x([0-9A-Fa-f]+)\]', token)
                    if hex_match:
                        hex_str = hex_match.group(1)
                        if len(hex_str) % 2 != 0: hex_str = "0" + hex_str
                        raw_pool += bytes.fromhex(hex_str)
                    else:
                        # 일반 텍스트는 인코딩 (여기서 '、'가 0xA1A2로 변환됨)
                        raw_pool += token.replace('\n', '\r\n').encode(self.encoding, errors='ignore')
                
                new_ptr_table.append(current_offset)
                new_data_pool += raw_pool + b'\x00'
                current_offset += len(raw_pool) + 1
            else:
                new_ptr_table.append(0)

        output_path = original_txd.replace(".txd", "_repacked.txd")
        with open(output_path, 'wb') as f:
            for ptr in new_ptr_table:
                f.write(struct.pack('<I', ptr))
            f.write(new_data_pool)
            
        print(f"▶ 리패킹 완료: {output_path}")

if __name__ == "__main__":
    tool = TXDTool()
    if len(sys.argv) < 3:
        print("사용법:")
        print("  추출: python txd_tool.py e 파일명.txd")
        print("  리팩: python txd_tool.py r 파일명.txd 파일명.txt")
    else:
        mode = sys.argv[1].lower()
        if mode == 'e': tool.extract(sys.argv[2])
        elif mode == 'r': tool.repack(sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else sys.argv[2] + ".txt"))