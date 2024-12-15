[Текст с заданиями](https://www.overleaf.com/read/vcqdvjkbjgzf)

Проверка решения:

1) Перейдите в папку с вашим решением (там, где есть папка utils)

```bash
cd <folder with utils folder>
```
<img width="260" alt="image" src="https://github.com/user-attachments/assets/61012e0c-356e-4574-8480-96ceca4c3f28" />

2) Запустите команду:

```bash
docker run --rm -it -v $PWD:/tmp joitandr/prac_recsys /bin/bash -c 'cp -r /tmp/* /prac_folder; cd /prac_folder; python test_submission.py; if [[ -f main_task_score.txt ]]; then mv main_task_score.txt ../tmp/; fi'
```

<img width="1442" alt="image" src="https://github.com/user-attachments/assets/52731e1d-bb07-4425-bdac-6225e12d2288" />

<img width="410" alt="image" src="https://github.com/user-attachments/assets/0878c72a-9bdb-4195-9d87-4c9ea7cad0bc" />

В результате у вас в папке <folder with utils folder> создастся файл `main_task_score.txt` с баллами за практическую