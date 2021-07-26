FROM gcr.io/deeplearning-platform-release/base-cpu

WORKDIR /

# Installs hypertune library
RUN pip install cloudml-hypertune imbalanced-learn scikit-learn==0.23.2 pandas==1.2.4 xgboost==1.2.1

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]