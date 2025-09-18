import zstandard as zstd
import pickle
import struct
import os

PATH = 'multi_tokens.zst'

cctx = zstd.ZstdCompressor(level=3)

def open_writer(path):
    """返回一个 callable：writer(obj) 把任意对象追加进同一个 zst 流"""
    f = open(path, 'wb')
    def write_obj(obj):
        pb = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        # 为每个对象创建独立的压缩帧
        compressed = cctx.compress(pb)
        f.write(struct.pack('<I', len(compressed)))  # 4 字节长度
        print(f"len(compressed) is {len(compressed)}")
        f.write(compressed)
    return write_obj, f

write_obj, f_out = open_writer(PATH)

# ---------- 模拟 100 段 ----------
for i in range(100):
    chunk = list(range(i*1_000_000, (i+1)*1_000_000))  # 每段 100 万数字
    write_obj(chunk)
    print(f'段 {i} 写入完成')

# 直接关闭，不需要 flush（因为每个对象都是独立压缩的）
f_out.close()
print('总文件大小:', os.path.getsize(PATH) / 1024 / 1024, 'MB')