import zstandard as zstd
import pickle
import struct
import io

PATH = 'multi_tokens.zst'

def iter_objects(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        while True:
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                break
            frame_size = struct.unpack('<I', len_bytes)[0]
            print(f"frame_size is {frame_size}")
            compressed = f.read(frame_size)
            
            # 检查是否读取到了完整的压缩数据
            if len(compressed) != frame_size:
                print(f"Warning: Expected {frame_size} bytes, got {len(compressed)} bytes")
                break

            # 方案 A：直接解压（每个对象都是独立的完整压缩帧）
            pb = dctx.decompress(compressed)
            yield pickle.loads(pb)

# ---------- 按需逐段读 ----------
total_elems = 0
for idx, chunk in enumerate(iter_objects(PATH)):
    total_elems += len(chunk)
    if idx % 10 == 0:
        print(f'段 {idx} 元素累计:', total_elems)
print('全部读完，总元素:', total_elems)