FROM public.ecr.aws/lambda/python:3.8

RUN pip install awscli --no-cache-dir

RUN chmod +x /var/task

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x ./kubectl

COPY ./*.py ${LAMBDA_TASK_ROOT}

COPY ./*.yaml ${LAMBDA_TASK_ROOT}

CMD ["create_rayjob.handler"]