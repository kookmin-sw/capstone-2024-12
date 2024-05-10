FROM public.ecr.aws/lambda/python:3.11

RUN pip install awscli requests --no-cache-dir

COPY main.py ${LAMBDA_TASK_ROOT}

RUN chmod +x /var/task

# x86_64
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x ./kubectl

CMD ["main.handler"]