import argparse
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("--password", help="Пароль от blackboxfunction.zip файла", type=str, default=None)
args = parser.parse_args()
password = args.password

if password is not None:
  with ZipFile('./blackboxfunction.zip') as zf:
    zf.extractall('./', pwd=bytes(password,'utf-8'))