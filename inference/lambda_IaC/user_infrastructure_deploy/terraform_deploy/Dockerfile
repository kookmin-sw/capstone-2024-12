FROM public.ecr.aws/lambda/python:3.12

COPY lambda.py ${LAMBDA_TASK_ROOT}
COPY ./terraform ${LAMBDA_TASK_ROOT}

RUN chmod +x /var/task

CMD ["lambda.handler"]