*Ссылка на текст практической:*

https://www.overleaf.com/read/tvmjbwbrrmyr

### Инструкция по проверке решения:

1. Устанавливаем докер
    
    [https://www.docker.com/get-started](https://www.docker.com/get-started)

2. Запускаем docker
3. Открываем терминал и пулим образ для проверки практической 
    
```angular2html
docker pull mlefmsu/prac_optimization
```
![Pull example](/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/docker_pull_example.png)

4. Проверяем, что этот образ скачался 

```angular2html
docker images
```
![Images example](/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/docker_images_ex.png)


5. Переходим в папку с вашей практической

![cd example](/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/cd.png)


6. Запускаем проверку
```angular2html
docker run --rm -it -v $PWD:/tmp mlefmsu/prac_optimization /bin/bash -c 'cp -r ./tmp/* ./prac_folder; cd ./prac_folder; bash test.sh; if [[ -f main_task_score.txt ]]; then cp main_task_score.txt ../tmp/; fi; if [[ -f blackbox_function_score.txt ]]; then cp blackbox_function_score.txt ../tmp/; fi'
```
![test example](/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/test.png)

7. После отработки кода в вашей директории появится 2 новых файла: main_task_score.txt и blackbox_function_score.txt

![new files example](/Users/antonandreytsev/Desktop/Machine-Learning-EF-MSU/pracs/msu_prac_optimization/pictures/new_files.png)
