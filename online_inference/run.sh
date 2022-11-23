if [ -z $MODEL_PATH ]
then
    export MODEL_PATH="model.pkl"
fi

if [ -z $TRANSFORMER_PATH ]
then
    export TRANSFORMER_PATH="transformer.pkl"
fi

if [[ ! -f $MODEL_PATH ]]
then
    gdown https://drive.google.com/.... --output=$MODEL_PATH
else
    echo "model already exists"
fi

if [[ ! -f $TRANSFORMER_PATH ]]
then
    gdown https://drive.google.com/... --output=$TRANSFORMER_PATH
else 
    echo "transformer aldeady exists"
fi

uvicorn main:app --reload --host 0.0.0.0 --port 8000