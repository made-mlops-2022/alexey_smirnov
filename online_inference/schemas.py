from typing import Literal

from pydantic import BaseModel, validator


class HeartDeasease(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator('age')
    def age_validator(cls, value):
        if value < 0 or value > 110:
            raise ValueError('wrong age value')
        return value

    @validator('trestbps')
    def trestbps_validator(cls, value):
        if value < 0 or value > 250:
            raise ValueError('wrong trestbps value')
        return value

    @validator('chol')
    def chol_validator(cls, value):
        if value < 0 or value > 700:
            raise ValueError('wrong chol value')
        return value

    @validator('thalach')
    def thalach_validator(cls, value):
        if value < 0 or value > 250:
            raise ValueError('wrong thalach value')
        return value

    @validator('oldpeak')
    def oldpeak_validator(cls, value):
        if value < 0 or value > 8:
            raise ValueError('wrong oldpeak value')
        return value
