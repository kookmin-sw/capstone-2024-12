FROM public.ecr.aws/lambda/python:3.12

COPY ./*.py ${LAMBDA_TASK_ROOT}

RUN chmod +x /var/task

RUN pip install --no-cache-dir pandas

CMD ["get_gpu_family_lambda.handler"]