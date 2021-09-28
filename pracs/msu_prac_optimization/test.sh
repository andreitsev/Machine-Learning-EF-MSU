#!/bin/bash

echo Тестируем submission.py
python3 /prac_folder/test_submission.py
echo

if [[ -f ./blackboxfunction.py ]]
then
  echo Тестируем blackbox_optimize.py
  python3 /prac_folder/blackbox_run.py
fi
