FROM python:3.8-slim

RUN pip install --no-cache-dir \
	numpy==1.19.5 \
    pandas==1.3.1 \
	fabulous==0.4.0 \
	typing==3.7.4.3 \
	argparse \
	scipy==1.5.2 \
	scikit-learn==0.24.2 \
	tqdm==4.61.1 \
	jupyter==1.0.0 \
	matplotlib==3.3.4 \
	seaborn==0.11.1 \
    category-encoders==2.2.2 \
    deepdiff==5.5.0

RUN mkdir /prac_folder

COPY . /prac_folder

EXPOSE 8989

CMD ["jupyter", "notebook", "--port=8989", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=joitandr1410"]