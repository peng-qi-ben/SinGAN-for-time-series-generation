# Read me

The code that needs to be run for the whole process of model training, generation, and testing is:

1. main_train.py
2. plot.py
3. 6+2test.py (For daily data)

## 1. Model Training

Run **main_train.py**.

The input of the model should be price data. To train with yield data, the parameters can be adjusted and in this case the output is also yield data. If the model is trained directly with price data, the output is a normalized price series.

The file will train the model and generate 1000 samples.

**The training results of the model** will be in the <TarinedModels> folder. This includes the trained saved models, and the Loss information for each layer.

**The results of the model generation** are in the <Output> folder.

## 2. Model testing and plot

Running the **plot.py** file first draws a plot of the original sequence; and draws images of several generated sequences. And it will also convert the original output sequence into the sequence that can be directly examined (for the 6+2test.py). 

**Parameter settings:**

1. **type**:  the name of the original training data file (without suffix); 

2. **param_singan** : can be selected from the output folder corresponding to the one to be examined and pasted directly.

**For Daily frequency data**:

 If you select daily frequency data in the parameters, it will first convert it into a price series to draw a series chart. The generated price series and yield series will be exported to the <daily_6+2_test> folder. The yield series can be then tested directly in the **6+2test.py** file. 

To test 8 indicators on daily frequency data, run the code in **6+2test.py** directly, with parameters consistent with those in **plot.py**. The results will be also in the <daily_6+2_test> folder.

**For monthly frequency data:**

The generated price series will be exported to the <monthly_fourier> folder. Then the Fourier transform can be used to check its periodicity, however, this part of the code cannot be disclosed.

## 3. Use the trained model

If <TraindeModels> has a corresponding trained model, run the **random_samples.py** file.





