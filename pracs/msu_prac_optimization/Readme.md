*Ссылка на текст практической:*

https://www.overleaf.com/read/tvmjbwbrrmyr

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
docker pull mlefmsu/prac_optimization
```

[comment]: <> (![Pull example]&#40;/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/docker_pull_example.png&#41;)

<img width="887" alt="docker_pull_example" src="https://user-images.githubusercontent.com/27732957/134806452-804b2cb6-40ee-4115-9e98-5290c8b3a295.png">

4. Проверяем, что этот образ скачался 

```angular2html
docker images
```

[comment]: <> (![Images example]&#40;/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/docker_images_ex.png&#41;)

<img width="768" alt="docker_images_ex" src="https://user-images.githubusercontent.com/27732957/134806497-549f3796-1406-41f3-bee4-2eacfb15ec38.png">


5. Переходим в папку с вашей практической

[comment]: <> (![cd example]&#40;/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/cd.png&#41;)

<img width="808" alt="cd" src="https://user-images.githubusercontent.com/27732957/134806526-e22a1b00-7426-43a3-9df5-f1a075add2fd.png">


6. Запускаем проверку

если у вас есть пароль от blackboxfunction.zip, то вот так:
```angular2html
docker run --rm -it -e password=<пароль от blackboxfunction.zip> -v $PWD:/tmp mlefmsu/prac_optimization /bin/bash -c 'cp -r ./tmp/* ./prac_folder; cd ./prac_folder; python3 unzip_blackboxfunction.py --password=$password; bash test.sh; if [[ -f main_task_score.txt ]]; then cp main_task_score.txt ../tmp/; fi; if [[ -f blackbox_function_score.txt ]]; then cp blackbox_function_score.txt ../tmp/; fi'                                                      
```
если нет, то вот так:
```angular2html
docker run --rm -it -v $PWD:/tmp mlefmsu/prac_optimization /bin/bash -c 'cp -r ./tmp/* ./prac_folder; cd ./prac_folder; python3 unzip_blackboxfunction.py; bash test.sh; if [[ -f main_task_score.txt ]]; then cp main_task_score.txt ../tmp/; fi; if [[ -f blackbox_function_score.txt ]]; then cp blackbox_function_score.txt ../tmp/; fi'                                                      
```

[comment]: <> (![test example]&#40;/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/test.png&#41;)

<img width="1417" alt="test" src="https://user-images.githubusercontent.com/27732957/134806579-8b980d80-a7bd-4a9c-b8cc-8db79fce6a7f.png">



7. После отработки кода в вашей директории появится 2 новых файла: main_task_score.txt и blackbox_function_score.txt

[comment]: <> (![new files example]&#40;/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/new_files.png&#41;)
<img width="1103" alt="new_files" src="https://user-images.githubusercontent.com/27732957/134806658-e95a7b2d-ac1d-432a-bc9f-0b86d82df4d4.png">
