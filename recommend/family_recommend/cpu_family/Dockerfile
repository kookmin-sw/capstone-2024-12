FROM public.ecr.aws/lambda/python:3.12

RUN pip install awscli

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

RUN chmod +x /var/task

CMD ["lambda_function.event_handler"]