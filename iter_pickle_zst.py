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

            # ① 方案 A：直接解压（需给 max_output_size）
            # pb = dctx.decompress(compressed, max_output_size=100_000_000)

            # ② 方案 B：流式解压（推荐）
            with dctx.stream_reader(io.BytesIO(compressed)) as reader:
                pb = reader.read()
            yield pickle.loads(pb)

# ---------- 按需逐段读 ----------
total_elems = 0
for idx, chunk in enumerate(iter_objects(PATH)):
    total_elems += len(chunk)
    if idx % 10 == 0:
        print(f'段 {idx} 元素累计:', total_elems)
print('全部读完，总元素:', total_elems)