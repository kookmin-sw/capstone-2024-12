FROM public.ecr.aws/lambda/python:3.12

COPY lambda.py ${LAMBDA_TASK_ROOT}
COPY ./terraform ${LAMBDA_TASK_ROOT}

ENV PATH="/var/task:${PATH}"

RUN chmod +x /var/task
RUN dnf install git -y
RUN dnf clean all

CMD ["lambda.handler"]