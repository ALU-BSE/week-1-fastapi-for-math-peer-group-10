from fastapi import FastAPI
import uvicorn 
import numpy as np 
import math
from pydantic import BaseModel

app = FastAPI()

class Matrix(BaseModel):
    matrix: list
 
#Initialise M & B as variable
M = np.random.rand(5,5)
B = np.random.rand(5,5)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



def f(x):
    pass
 
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy
#Return 

#initialize x as a 5 * 5 matrix

#Make a call to the function

#Recreate the function with the sigmoid Function

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

