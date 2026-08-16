import time, threading
from config.settings import csv_list, get_settings
from ingest.providers.bybit import BybitBarProvider

syms = csv_list(get_settings().ingest_default_crypto_symbols)
p = BybitBarProvider()
out = open("/tmp/bybit_probe.txt", "w")


def probe(sym):
    t = time.time()
    try:
        rows = p._fetch_symbol(sym, 0, 180)
        out.write(f"OK {sym} bars={len(rows)} ms={round((time.time() - t) * 1000)}\n")
    except Exception as e:
        out.write(
            f"ERR {sym} {type(e).__name__} {str(e)[:80]} ms={round((time.time() - t) * 1000)}\n"
        )
    out.flush()


for s in syms:
    th = threading.Thread(target=probe, args=(s,))
    th.start()
    th.join(15)
    if th.is_alive():
        out.write(f"HANG {s} >15s\n")
        out.flush()
out.write("DONE\n")
out.close()
