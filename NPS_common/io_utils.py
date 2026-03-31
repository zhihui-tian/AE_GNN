import subprocess
import os
import tempfile
from contextlib import contextmanager

def co(instr, split=False):
    out=subprocess.Popen(instr, stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()[0]
    return out.split('\n') if split else out


@contextmanager
def temp_txt_file(data):
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wt')
    temp.write(data)
    temp.close()
    try:
        yield temp.name
    finally:
        os.unlink(temp.name)


