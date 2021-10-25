*Ссылка на текст практической:*

https://www.overleaf.com/read/yrwdkbnyywtv


### Инструкция по проверке решения:

#### На Windows:

[Инструкция](https://www.evernote.com/shard/s596/client/snv?noteGuid=ec85951e-7bd1-b108-b7cc-7a1fea9ccb23&noteKey=4ee1a4a181642b77ff16ab69dd0b97e4&sn=https%3A%2F%2Fwww.evernote.com%2Fshard%2Fs596%2Fsh%2Fec85951e-7bd1-b108-b7cc-7a1fea9ccb23%2F4ee1a4a181642b77ff16ab69dd0b97e4&title=%25D0%25A0%25D0%25B0%25D0%25B1%25D0%25BE%25D1%2582%25D0%25B0%2B%25D1%2581%2BDocker%2B%25D1%2587%25D0%25B5%25D1%2580%25D0%25B5%25D0%25B7%2BWSL)

Альтернативный вариант:

[Использовать gcloud shell](https://cloud.google.com/blog/products/it-ops/introducing-the-ability-to-connect-to-cloud-shell-from-any-terminal)

[Использовать play-with-docker](https://labs.play-with-docker.com/)

#### На Linux/Mac:

1. Устанавливаем докер
    
    [https://www.docker.com/get-started](https://www.docker.com/get-started)

2. Запускаем docker
3. Открываем терминал и пулим образ для проверки практической 
    
```angular2html
docker pull mlefmsu/prac_trees
```

<img width="847" alt="pull" src="https://user-images.githubusercontent.com/27732957/138518669-811c2d71-e924-46ce-8d21-1ebdd7b16326.png">

4. Проверяем, что этот образ скачался 

```angular2html
docker images
```

<img width="770" alt="images" src="https://user-images.githubusercontent.com/27732957/138518779-cbebf228-9add-4f23-b554-91e85e27765d.png">

5. Переходим в папку с вашей практической

<img width="685" alt="cd" src="https://user-images.githubusercontent.com/27732957/138518834-2150d072-ee72-4f1f-a29e-4d872bdc2650.png">

6. Запускаем проверку

если у вас есть пароль от blackboxfunction.zip, то вот так:
```angular2html
docker run --rm -it -e password=<пароль от private_tests.zip> -v $PWD:/tmp mlefmsu/prac_trees /bin/bash -c 'cp -r /tmp/* /prac_folder; cd /prac_folder; python3 unzip_private_tests.py --password=$password; python3 test_submission.py; if [[ -f main_task_score.txt ]]; then mv main_task_score.txt ../tmp/; fi'                                                      
```
если нет, то вот так:
```angular2html
docker run --rm -it -v $PWD:/tmp mlefmsu/prac_trees /bin/bash -c 'cp -r /tmp/* /prac_folder; cd /prac_folder; python3 unzip_private_tests.py; python3 unzip_private_tests.py; if [[ -f main_task_score.txt ]]; then mv main_task_score.txt ../tmp/; fi'                                                      
```

<img width="1337" alt="docker run" src="https://user-images.githubusercontent.com/27732957/138518933-ea212144-97ef-4e97-9a0e-dd1a381f1e03.png">


7. После отработки кода в вашей директории появится 1 новый файл: main_task_score.txt

<img width="566" alt="result" src="https://user-images.githubusercontent.com/27732957/138519211-a34d0751-e70c-4253-b9b5-160b84cdca11.png">