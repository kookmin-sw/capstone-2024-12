FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir pandas

COPY ./*.py ${LAMBDA_TASK_ROOT}

CMD ["lambda.handler"]