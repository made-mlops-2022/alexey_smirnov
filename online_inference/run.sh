if [ -z $MODEL_PATH ]
then
    export MODEL_PATH="output_model.pkl"
fi

if [ -z $TRANSFORMER_PATH ]
then
    export TRANSFORMER_PATH="transformer.pkl"
fi

if [[ ! -f $MODEL_PATH ]]
then
    gdown https://drive.google.com/file/d/1xPq7zv8cH0lVfKPOflhEYuoYGVU37yi-/view?usp=sharing --fuzzy --output=$MODEL_PATH
else
    echo "model already exists"
fi

if [[ ! -f $TRANSFORMER_PATH ]]
then
    gdown https://drive.google.com/file/d/1Vsa6G9dqbUnZ6QPwU3bnQphot287-jxX/view?usp=sharing --fuzzy --output=$TRANSFORMER_PATH
else 
    echo "transformer already exists"
fi

uvicorn main:app --reload --host 0.0.0.0 --port 8000