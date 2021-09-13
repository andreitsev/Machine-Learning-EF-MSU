FROM python:3.8-slim

RUN pip install --no-cache-dir \
	numpy==1.19.5 \
	fabulous==0.4.0 \
	typing==3.7.4.3 \
	argparse \
	scipy==1.5.2 \
	scikit-learn==0.24.2 \
	tqdm==4.61.1

RUN mkdir /prac_folder

COPY . /prac_folder

CMD ["bash", "/prac_folder/test.sh"]