from gguf import GGUFReader
import re
for lbl, p in [
    ('omn', r'D:\_MODELS\LiquidAI\LFM3-Omni-2B-GGUF\LFM3-Omni-2B-Q8_0.gguf'),
]:
    r = GGUFReader(p)
    # find all lfm3.* metadata
    print(f'== {lbl} ==')
    for k in sorted(r.fields.keys()):
        if k.startswith('lfm3.') and ('sliding' in k or 'shortconv' in k or 'window' in k or 'block' in k or 'rope' in k):
            try:
                v = r.fields[k].data
                print(f'  {k} = {v}')
            except Exception as e:
                print(f'  {k} = <err {e}>', v if 'v' in locals() else '', )