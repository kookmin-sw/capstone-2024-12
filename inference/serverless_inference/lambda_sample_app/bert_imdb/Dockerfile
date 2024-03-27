FROM public.ecr.aws/lambda/python:3.10
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY ./bert_imdb ${LAMBDA_TASK_ROOT}/bert_imdb
COPY ./app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.lambda_handler" ]